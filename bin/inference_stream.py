from __future__ import print_function

import builtins
import datetime
import time
import math
import argparse
import threading
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator

import soundfile as sf
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as k

from bin.inference import audioEncoderProcessor
from models.utils import print_outputs
from web.queue import PCMQueue, ThreadSafeQueue
from models.pipeline import inferencePipeline
from models.decoder.llm2tts import llm2TTS


def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    original_print(f"[{current_time}]", *args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description="Freeze-Omni-Inference-stream")
    parser.add_argument("--model_path", required=True, help="model_path to load")
    parser.add_argument("--llm_path", required=True, help="llm_path to load")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--input_wav", required=True, help="input wav")
    parser.add_argument("--output_wav", required=True, help="output wav")

    args = parser.parse_args()
    print(args)
    return args


class GlobalVars:
    """
    multi turn conversation interal speech dialogue outputs for listen and speak
    """

    speech_dialogue_outputs = {}

    @staticmethod
    def deepcopy_outputs():  # cp for write
        return deepcopy(GlobalVars.speech_dialogue_outputs)


@dataclass
class PCMStatChunk:
    status: str  # sl(start listen), cl(continue listen), el(end listen)
    data: torch.Tensor  # chunk tensor

    def __post_init__(self):
        if self.status not in ["sl", "cl", "el"]:
            raise ValueError("status must be one of: 'sl','cl','el'")

    def __str__(self):
        return (
            f"status:{self.status} data shape:{self.data.shape if self.data is not None else None}"
        )


@dataclass
class Listen2SpeakFrame:
    """
    outputs have stat, so need pass it from listen to speak
    """

    outputs: dict

    def __str__(self):
        return f"outputs:{print_outputs(self.outputs)}"


@dataclass
class GenTTSFrame:
    text: str
    data: np.ndarray  # a numpy array of dtype np.float32

    def __str__(self):
        return f"text:{self.text} numpy data:{self.data.shape if self.data is not None else None}"


class PCMListener:
    def __init__(
        self,
        outputs_queue: ThreadSafeQueue,  # outputs -> out queue
        pipeline: inferencePipeline,
        cache_history_size: int = 10,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.pcm_stat_chunk_queue = ThreadSafeQueue()  # pcm data -> in queue

        self.outputs_queue = outputs_queue

        # stream chunk to encoder
        self.audio_processor = audioEncoderProcessor()
        # encoder and audio llm
        self.pipeline = pipeline

        # pre status system prompt, outputs stat: pre -> sl
        self.outputs = self.pipeline.speech_dialogue(None, stat="pre", role=system_prompt)
        # print(f"pre-> outputs:[{print_outputs(self.outputs)}]")
        GlobalVars.speech_dialogue_outputs = self.outputs

        # chunck feat history cache also use ring buffer to do :)
        self.history = torch.zeros(
            [
                cache_history_size,
                self.audio_processor.chunk_size + self.audio_processor.chunk_overlap,
                self.audio_processor.feat_dim,
            ]
        )

        # start listen thread
        self.listen_thread = threading.Thread(target=self.listen, args=())
        self.stop_listen = False
        self.listen_thread.start()

    def stop(self):
        self.stop_listen = True
        self.listen_thread.join(timeout=3)

    def send(self, pcm_items: torch.Tensor | None, status: str):
        if pcm_items is None:
            item = PCMStatChunk(status=status, data=None)
            self.pcm_stat_chunk_queue.put(item)
        else:
            item = PCMStatChunk(status=status, data=pcm_items)
            self.pcm_stat_chunk_queue.put(item)

    def history_buffering_strategy(self, input_chunk: torch.Tensor) -> torch.Tensor:
        # cache fbank feature (input_chunk)
        # << 1
        self.history[:-1] = self.history[1:].clone()
        # last history = input chunk
        self.history[-1:] = input_chunk
        # start listen to # copy last 6 chunks
        feature_last_chunk = self.history[-6:].unsqueeze(1)
        return feature_last_chunk

    def llm_prefill(self, status: str, feature: torch.Tensor, outputs: dict, is_first_pack=False):
        """
        Prefills the LLM of speech dialogue system using speech.

        Parameters:
        - status: the current state of the audio input.
        - feature: audio feature tensor.
        - outputs (dict): A dictionary containing the current state of the dialogue system.
        - is_first_pack (bool, optional): Indicates if the current input packet is the first one in a new conversation
        """

        if status == "sl":
            # Satge1: start listen
            # stat will be auto set to 'cl' after Stage1
            outputs = self.pipeline.speech_dialogue(feature, **outputs)
            return outputs
        if status == "el":
            print("status end listen. start to speak")
            return outputs

        if status == "cl":
            if outputs["stat"] == "cl" or outputs["stat"] == "sl":
                # Stage2: continue listen
                # stat will be auto set to 'ss' when endpoint is detected
                # print("output stat continue listen")
                outputs = self.pipeline.speech_dialogue(feature, **outputs)
            if is_first_pack:
                outputs["stat"] = "cl"
            if outputs["stat"] == "el":
                print("output stat end listen. Detect invalid break")
            if outputs["stat"] == "ss":
                # print(f"start speak. start to speak")
                pass
        return outputs

    def listen(self):
        """
        chunk status from VAD
        """
        print("Start listening")
        while True:
            if self.stop_listen:
                print("Stop listening")
                break
            stat_chunk: PCMStatChunk = self.pcm_stat_chunk_queue.get()
            if stat_chunk is None:
                time.sleep(0.01)
                continue
            # print(f"Received PCM stat chunk: {stat_chunk}")

            # if self.outputs['stat'] =="sl":
            #    print(f"stat_chunk data:{stat_chunk.data}")

            if stat_chunk.status == "sl":
                fbank_feature = self.audio_processor.process(stat_chunk.data)
                self.outputs = (
                    GlobalVars.deepcopy_outputs()
                )  # for next turn conversation outputs from speak where in gloab var
                # print(f"sl-> outputs:{print_outputs(self.outputs)}")
                self.outputs["adapter_cache"] = None
                self.outputs["encoder_cache"] = None
                self.outputs["pe_index"] = 0
                self.outputs["stat"] = "sl"
                self.outputs["last_id"] = None
                if "text" in self.outputs:
                    del self.outputs["text"]
                if "hidden_state" in self.outputs:
                    del self.outputs["hidden_state"]

                self.outputs = self.llm_prefill("sl", fbank_feature, self.outputs)

            elif stat_chunk.status == "cl":
                # print(f"cl in-> outputs:{print_outputs(self.outputs)}")
                fbank_feature = self.audio_processor.process(stat_chunk.data)
                # if self.outputs['stat'] =="sl":
                #    print(f"fbank:{fbank_feature}")
                self.outputs = self.llm_prefill(stat_chunk.status, fbank_feature, self.outputs)
                # print(f"cl out-> outputs:{print_outputs(self.outputs)}")
                if self.outputs["stat"] == "ss":
                    pass
                else:
                    self.outputs["stat"] = "cl"
            elif stat_chunk.status == "el":
                self.audio_processor.reset()

                # print(f"el-> outputs:[{print_outputs(self.outputs)}]")
                self.outputs["adapter_cache"] = None
                self.outputs["encoder_cache"] = None
                self.outputs["pe_index"] = 0
                self.outputs["last_id"] = None
                self.outputs["stat"] = "ss"

            if self.outputs["stat"] == "ss":
                frame = Listen2SpeakFrame(self.outputs)
                # print(f"start to speak, send frame:[{frame}]")
                self.outputs_queue.put(frame)


@dataclass
class TTSSpeakerArgs:
    # https://huggingface.co/VITA-MLLM/Freeze-Omni/blob/main/checkpoints/server.json
    # decoder(LLM2TTSCodecAR)
    # NAR llama transformer decoder pre_nn_forward -> NAR llama transformer decoder kv_cache_prefix_forward -> AR llama transformer decoder transformer_infer

    # llama transformer decoder
    decoder_top_k: int = 2
    decoder_penalty_window_size: int = -1  # <0 no penalty window
    decoder_penalty: float = 1.1

    # codec decoder
    decoder_first_chunk_size: int = 20
    decoder_chunk_size: int = 40
    decoder_chunk_overlap_size: int = 10

    # find_min_sum_index
    decoder_N: int = 2401
    decoder_seg_threshold_first_pack: float = 0.1
    decoder_seg_threshold: float = 0.015


class TTSSpeaker:
    def __init__(
        self,
        args: TTSSpeakerArgs,
        outputs_in_queue: ThreadSafeQueue,  # out queue -> speak
        pipeline: inferencePipeline,
        tts: llm2TTS,
    ) -> None:
        self.args = args
        self.outputs_in_queue = outputs_in_queue
        self.pipeline = pipeline
        self.tts = tts

        self.reset()

        # speak thread
        self.speak_thread = threading.Thread(target=self.speak, args=())
        self.speak_thread.start()

    def reset(self):
        self.stop_speak = False
        self.is_generate = False
        self.whole_text = ""

        self.tts_over = False
        self.tts_over_time = 0
        self.tts_data = ThreadSafeQueue()

        self.stop_tts = False

    def print(self):
        print("stop_speak:", self.stop_speak)
        print("is_generate:", self.is_generate)
        print("whole_text:", self.whole_text)
        print("tts_over:", self.tts_over)
        print("tts_over_time:", self.tts_over_time)

    @property
    def gen_text(self):
        """Get the whole text."""
        return self.whole_text

    def stop(self):
        """Stop the speak thread."""
        self.stop_speak = True
        self.speak_thread.join()

    def decoder(
        self, cur_hidden_state: list[torch.Tensor], cur_text: str, generate_num: int
    ) -> int:
        """
        Decodes the current hidden state and text to generate audio segments using speech decoder.

        Parameters:
        - cur_hidden_state (list of torch.Tensor): The current hidden state of the language model.
        - cur_text (str): The current text to be synthesized.
        - generate_num (int): The number of audio segments generated

        Returns:
        - int: The updated number of audio segments generated.
        """
        hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
        cur_text_procced = self.pipeline.post_process(cur_text)
        print("Synthesis: ", [cur_text_procced])
        embeddings = self.pipeline.model.llm_decoder.model.embed_tokens(
            torch.tensor(self.pipeline.model.tokenizer.encode(cur_text_procced)).cuda()
        )
        codec_chunk_size = self.args.decoder_first_chunk_size
        codec_padding_size = self.args.decoder_chunk_overlap_size
        seg_threshold = self.args.decoder_seg_threshold_first_pack
        if generate_num != 0:
            codec_chunk_size = self.args.decoder_chunk_size
            seg_threshold = self.args.decoder_seg_threshold
        for seg in self.tts.run(
            embeddings.reshape(-1, 896).unsqueeze(0),
            self.args.decoder_top_k,
            hidden_state_output.reshape(-1, 896).unsqueeze(0),
            codec_chunk_size=codec_chunk_size,
            codec_padding_size=codec_padding_size,
            penalty_window_size=self.args.decoder_penalty_window_size,
            penalty=self.args.decoder_penalty,
            N=self.args.decoder_N,
            seg_threshold=seg_threshold,
        ):
            if generate_num == 0:
                try:
                    split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                    seg = seg[:, :, split_idx:]
                except Exception:
                    print("Do not need to split")
                    pass
            generate_num += 1
            if self.tts_over:
                self.tts_data.clear()
                self.whole_text = ""
                self.tts_data.put(GenTTSFrame(text="", data=None))
                break
            frame = GenTTSFrame(text=cur_text, data=seg.squeeze().float().cpu().numpy())
            self.tts_data.put(frame)
        return generate_num

    def get_tts_data(self) -> Generator[GenTTSFrame, None, None]:
        """
        get tts bytes data
        """
        while True:
            if self.stop_speak:
                print("Stop speak so break get tts data")
                break
            output_data = self.tts_data.get()
            if output_data is not None:
                # print("Get TTS data")
                # yield output_data.astype(np.int16).tobytes()
                yield output_data
            else:
                yield None

    def speak(self):
        """
        Generates speech dialogue output based on the current state
        """
        while True:
            if self.stop_speak:
                print("Stop speak")
                break
            frame: Listen2SpeakFrame = self.outputs_in_queue.get()
            if frame is None:
                time.sleep(0.01)
                continue

            outputs = frame.outputs
            # print(f"speak get-> outputs:[{print_outputs(outputs)}]")
            # Stage3: start speak
            self.is_generate = True
            outputs = self.pipeline.speech_dialogue(None, **outputs)
            # print(f"ss-> outputs:[{print_outputs(outputs)}]")

            cur_hidden_state = []
            cur_hidden_state.append(outputs["hidden_state"])

            # Stage4: contiune speak until stat is set to 'sl'
            # use 'stop' to interrupt generation, stat need to be manually set as 'sl'
            stop = False
            cur_text = ""
            last_text = ""
            generate_num = 0
            while True:
                if self.stop_speak:
                    break
                if len(outputs["past_tokens"]) > 500:
                    stop = True
                if stop:
                    break
                del outputs["text"]
                del outputs["hidden_state"]
                outputs = self.pipeline.speech_dialogue(None, **outputs)
                # print(f"sc-> outputs:[{print_outputs(outputs)}]")
                if outputs["stat"] == "cs":
                    cur_hidden_state.append(outputs["hidden_state"])
                    if "�" in outputs["text"][len(last_text) :]:
                        continue
                    self.whole_text += outputs["text"][len(last_text) :]
                    cur_text += outputs["text"][len(last_text) :]
                    # print(self.whole_text])
                    if generate_num == 0 or (len(cur_hidden_state) >= 20):
                        suffix_list = [",", "，", "。", "：", "？", "！", ".", ":", "?", "!", "\n"]
                    else:
                        suffix_list = ["。", "：", "？", "！", ".", "?", "!", "\n"]
                    if outputs["text"][len(last_text) :].endswith(tuple(suffix_list)) and (
                        len(cur_hidden_state) >= 4
                    ):
                        if (
                            outputs["text"][len(last_text) :].endswith(".")
                            and last_text[-1].isdigit()
                        ):
                            pass
                        else:
                            if not self.tts_over:
                                if len(cur_hidden_state) > 0:
                                    generate_num = self.decoder(
                                        cur_hidden_state, cur_text, generate_num
                                    )
                                    cur_text = ""
                                    cur_hidden_state = []
                    last_text = outputs["text"]
                else:
                    break
            if not self.tts_over:
                if len(cur_hidden_state) != 0:
                    generate_num = self.decoder(cur_hidden_state, cur_text, generate_num)
                    cur_text = ""
            self.is_generate = False
            outputs["stat"] = "sl"
            outputs["last_id"] = None
            GlobalVars.speech_dialogue_outputs = deepcopy(outputs)
            print(self.whole_text)
            self.tts_data.put(GenTTSFrame(text="", data=None))

    def interrupt(self):
        self.stop_speak = True
        self.tts_over = True
        while True:
            time.sleep(0.01)
            if self.is_generate is False:
                self.stop_speak = False
                while True:
                    time.sleep(0.01)
                    if self.tts_data.is_empty():
                        self.whole_text = ""
                        self.tts_over = False
                        self.tts_over_time += 1
                        break
                break


def inference_stream(listener: PCMListener, speaker: TTSSpeaker, configs):
    """
    Perform inference for a speech dialogue system.
    - 流式语音输入通过语音编码器形成chunk-wise特征，然后通过适配器连接到LLM。
    - LLM生成隐藏状态和文本标记，在块分割后分别以块的形式发送到NAR前缀语音解码器和NAR语音解码器。
    - 最后，AR语音解码器将生成的令牌发送到语音令牌FIFO中，流式编解码器根据固定的语音令牌块大小从FIFO生成流式语音输出。

    Parameters:
    - listener: listen pcm data(chunk) to asr with pipeline(encoder and adpter),
        - detail status: sl, cl, el, ss
    - speaker: tts speaker with pipeline(audio llm) and decoder (NAR decoder AR decoder and codec decoder)
        - detail status: ss, cs
    - configs: Input args. (argparse)

    Returns:
    - None
    """
    wav, fs = sf.read(configs.input_wav)
    wav = torch.tensor(wav)
    if fs != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav.float())
        fs = 16000

    # like io_uring
    chunk_size = listener.audio_processor.get_chunk_size()
    wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
    wav_input[: wav.shape[0]] = wav
    for i in range(0, wav_input.shape[0], chunk_size):
        print("--->", wav_input.shape, wav.shape, wav_input[i : i + chunk_size].shape)
        status = "cl"
        if i == 0:
            status = "sl"
        # send pcm data with status
        listener.send(wav_input[i : i + chunk_size], status)
    listener.send(None, "el")

    wavs = []
    # get tts speak data
    for item in speaker.get_tts_data():
        if item:
            print(item)
            if item.data is None:
                break
            else:
                wavs.append(item.data)
        else:
            time.sleep(0.01)  # yield thread

    sf.write(configs.output_wav, np.concatenate(wavs, -1), 24000)
    print(f"write to {configs.output_wav}")

    listener.stop()
    speaker.stop()


if __name__ == "__main__":
    # change print function to add time stamp
    original_print = builtins.print
    builtins.print = custom_print

    configs = get_args()

    # encoder and audio llm
    pipeline = inferencePipeline(configs)
    # decoder
    tts = llm2TTS(configs.model_path)
    # listen -> gen_queue -> speak
    gen_queue = ThreadSafeQueue()
    # listen
    listener = PCMListener(gen_queue, pipeline)
    # speak
    speaker = TTSSpeaker(TTSSpeakerArgs(), gen_queue, pipeline, tts)

    inference_stream(listener, speaker, configs)


# format: ruff format bin/inference_stream.py
