from __future__ import print_function

import argparse
import torch
import torchaudio
import datetime
import builtins
import math

import soundfile as sf
import numpy as np
import torchaudio.compliance.kaldi as k

from models.utils import print_outputs
from models.pipeline import inferencePipeline
from models.decoder.llm2tts import llm2TTS

def get_args():
    parser = argparse.ArgumentParser(description='Freeze-Omni')
    parser.add_argument('--model_path', required=True, help='model_path to load')
    parser.add_argument('--llm_path', required=True, help='llm_path to load')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--input_wav', required=True, help='input wav')
    parser.add_argument('--output_wav', required=True, help='output wav')

    args = parser.parse_args()
    print(args)
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)


class audioEncoderProcessor:
    def __init__(self, chunk_size = 16):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        self.reset()
    
    def get_chunk_size(self):
        return self.CHUNK
    
    def reset(self):
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
    
    def fbank_shift(self, sample_data):
        # fbank feature shift
        self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
        self.input_sample[:, self.frame_overlap:, :] = sample_data
    
    def chunk_data_shift(self, xs):
        # chunk feature shift
        self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
        self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)
    
    def process(self,
                audio: torch.Tensor):
        """
        # 1. Converts the input audio tensor to the appropriate format.
        # 2. Computes the filter bank features (fbank) for the audio.
        # 3. Updates the input chunk and history based on the new audio segment.
        """
        with torch.no_grad():
            sample_data = audio.clone().reshape(1, -1, 1)[:, :, :1] * 32768
            self.fbank_shift(sample_data)
            # use kaldi api to compute fbank
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                         frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.chunk_data_shift(xs)
        return self.input_chunk.clone()

def decoder(cur_hidden_state, pipeline, cur_text, tts, codec_chunk_size, codec_padding_size, decoder_topk, wav):
    """
    Decodes the current hidden state and text to generate audio segments using speech decoder.
    """
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = pipeline.post_process(cur_text)
    print("Synthesis: ", [cur_text_procced])
    embeddings = pipeline.model.llm_decoder.model.embed_tokens(
                    torch.tensor(pipeline.model.tokenizer.encode(cur_text_procced)).cuda()
                    )
    for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), decoder_topk,
                        hidden_state_output.reshape(-1, 896).unsqueeze(0), 
                        codec_chunk_size, codec_padding_size):
        wav.append(seg)

def inference(pipeline:inferencePipeline, audio_processor:audioEncoderProcessor, tts:llm2TTS, configs):
    """
    Perform inference for a speech dialogue system.

    Parameters:
    - pipeline: Speech dialogue pipeline.
    - audio_processor: Processes raw audio data into a format suitable for the pipeline.
    - tts: The speech decoder moudule.
    - configs: Input args.

    Returns:
    - None
    """
    wav, fs = torchaudio.load(configs.input_wav)
    if fs != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav)
        fs = 16000
    wav = wav.reshape(-1)

    #wav, fs = sf.read(configs.input_wav)
    #wav = torch.tensor(wav)
    #if fs != 16000:
    #    wav = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(wav.float())
    #    fs = 16000
    print("--->",wav.shape)
    
    codec_chunk_size = 40
    codec_padding_size = 10
    decoder_topk = 2

    # Satge0: preprocess
    # set system role, stat will be set to 'sl'  
    stat = 'pre'
    outputs = pipeline.speech_dialogue(None, stat=stat, role="You are a helpful assistant.")
    print(f"pre-> outputs:[{print_outputs(outputs)}]")
    chunk_size = audio_processor.get_chunk_size()
    
    # Satge1: start listen
    # stat will be auto set to 'cl' after Stage1
    wav_input = torch.zeros(math.ceil(wav.shape[0] / chunk_size) * chunk_size)
    wav_input[:wav.shape[0]] = wav
    for i in range(0, wav_input.shape[0], chunk_size):
        print("--->",wav_input.shape, wav.shape,wav_input[i:i+chunk_size].shape)
        print(f"cl in-> outputs:{print_outputs(outputs)}")
        if outputs['stat'] =="sl":
            print(f"stat_chunk data:{wav_input[i:i+chunk_size]}")
        fbank = audio_processor.process(wav_input[i:i+chunk_size])
        if outputs['stat'] =="sl":
            print(f"fbank:{fbank}")
        outputs = pipeline.speech_dialogue(fbank, **outputs)
        print(f"cl out-> outputs:{print_outputs(outputs)}")
        outputs['stat'] = 'cl'
    audio_processor.reset()
        
    print("listen",outputs.keys())
    print(f"listen-> outputs:[{print_outputs(outputs)}]")
    outputs['adapter_cache'] = None
    outputs['encoder_cache'] = None
    outputs['pe_index'] = 0
    outputs['stat'] = 'ss'
    print(f"speak get-> outputs:[{print_outputs(outputs)}]")

    # Stage3: start speak
    outputs = pipeline.speech_dialogue(None, **outputs)
    print(f"ss-> outputs:[{print_outputs(outputs)}]")
    cur_hidden_state = []
    cur_hidden_state.append(outputs['hidden_state'])

    whole_text = ''
    last_text = ''
    cur_text = ''
    wav = []
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'  
    stop = False
    while True:
        if len(outputs['past_tokens']) > 128:
            stop = True
        if stop:
            break
        del outputs['text']
        del outputs['hidden_state']
        outputs = pipeline.speech_dialogue(None, **outputs)
        print(f"sc-> outputs:[{print_outputs(outputs)}]")
        if outputs['stat'] == 'cs':
            cur_hidden_state.append(outputs['hidden_state'])
            whole_text += outputs['text'][len(last_text):]
            cur_text += outputs['text'][len(last_text):]
            suffix_list = ["。", "：", "？", "！", ".", "?","!", "\n"]
            if outputs['text'][len(last_text):].endswith(tuple(suffix_list)):
                if outputs['text'][len(last_text):].endswith(".") and last_text[-1].isdigit():
                    pass
                else:
                    if len(cur_hidden_state) > 0:
                        decoder(cur_hidden_state, pipeline, cur_text, tts, 
                                codec_chunk_size, codec_padding_size, decoder_topk, wav)
                        cur_hidden_state = []
                        print(f"cur_text:{cur_text}")
                    cur_text = ""
        if outputs['stat'] == 'sl':
            break
        #print(outputs['text'])
        last_text = outputs['text']
    if len(cur_hidden_state) != 0:
        print(f"cur_text:{cur_text}")
        decoder(cur_hidden_state, pipeline, cur_text, tts, 
                codec_chunk_size, codec_padding_size, decoder_topk, wav)

    sf.write(configs.output_wav, torch.cat(wav, -1).squeeze().float().cpu().numpy(), 24000)
    outputs['stat'] = 'sl'
    outputs['last_id'] = None
    print(whole_text)

if __name__ == '__main__':
    # change print function to add time stamp
    original_print = builtins.print
    builtins.print = custom_print

    configs = get_args()
    # encoder and audio llm
    pipeline = inferencePipeline(configs)
    # decoder
    tts = llm2TTS(configs.model_path)
    # stream chunk to encoder
    audio_processor = audioEncoderProcessor()
    inference(pipeline, audio_processor, tts, configs)
