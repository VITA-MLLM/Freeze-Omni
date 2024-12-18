import json
import torch
import threading
import queue
import subprocess
import concurrent.futures

import numpy as np
import torchaudio.compliance.kaldi as k

from silero_vad.model import load_silero_vad
from silero_vad.utils_vad import VADIterator

class VAD:
    """
    首先使用声学 VAD 模块来检测流式演讲的起点。
    当VAD被触发时，语音流将被逐块发送到Freeze-Omni，并在LLM最后一层之后添加一个额外的分类层来预测不同的状态。
    这里定义了三种状态:
    - 状态0表示当前LLM可以继续接收语音，
    - 状态1或2表示当前块是语音结束。
        - 状态1表示用户将中断对话，LLM将执行新的生成阶段，
        - 状态2表示无需中断对话。
        这两种状态都将停止向 Freeze-Omni 发送语音流并重置 VAD 模块。
    """
    def __init__(self, cache_history=10, last_chunk_size=6):
        self.chunk_size = 16
        self.chunk_overlap = 3
        self.feat_dim = 80
        self.frame_size = 400
        self.frame_shift = 160
        self.frame_overlap = self.frame_size - self.frame_shift
        self.CHUNK = self.frame_shift * self.chunk_size
        assert cache_history >= last_chunk_size, "cache_history must >= last_chunk_size"
        self.cache_history = cache_history
        self.last_chunk_size = last_chunk_size
        self.in_dialog = False

        with torch.no_grad():
            self.load_vad()
            self.reset_vad()
    
    def get_chunk_size(self):
        return self.CHUNK

    def load_vad(self):
        self.vad_model = load_silero_vad()
        self.vad_model.eval()
        # generate vad itertator
        self.vad_iterator = VADIterator(self.vad_model, 
                                        threshold=0.8, 
                                        sampling_rate=16000, 
                                        min_silence_duration_ms=2000, 
                                        speech_pad_ms=30)
        self.vad_iterator.reset_states()

    def reset_vad(self):
        # reset all parms
        self.input_chunk = torch.zeros([1, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
        # chunck feat history cache also use ring buffer to do :)
        self.history = torch.zeros([self.cache_history, self.chunk_size + self.chunk_overlap, self.feat_dim])
        self.vad_iterator.reset_states()
        self.in_dialog = False
    
    def run_vad_iterator(self, audio):
        speech_dict_out = None
        # split into chunk with 512
        for i in range(len(audio) // 512):
            speech_dict = self.vad_iterator(audio[i * 512: (i + 1) * 512], return_seconds=True)
            if speech_dict is not None:
                speech_dict_out = speech_dict
        return speech_dict_out
    
    def predict(self,
                audio: np.ndarray):
        """
        Predict the Voice Activity Detection (VAD) status and return related features.

        Parameters:
        - audio (torch.Tensor): A 1D or 2D tensor representing the input audio signal.

        Returns:
        - return_dict (dict): A dictionary containing the VAD status and related features.
            - 'status' (str): The current VAD status, which can be 'sl' (speech start), 
                              'cl' (speech continue), or 'el' (speech end).
            - 'feature_last_chunk' (list of list of float): The feature of the last chunks.
            - 'feature' (list of list of float): The feature of the current chunk of audio.
            - 'history_feature' (list of list of list of float): The cached features of previous chunks.
        
        """

        # 1. Converts the input audio tensor to the appropriate format.
        # 2. Computes the filter bank features (fbank) for the audio.
        # 3. Updates the input chunk and history based on the new audio segment.
        # 4. Determines the VAD status by running the VAD iterator on the audio.
        # 5. Populates the return dictionary with the VAD status and related features.

        return_dict = {}
        return_dict['status'] = None
        with torch.no_grad():
            # get fbank feature
            audio = torch.tensor(audio)
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32768
            self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
            self.input_sample[:, self.frame_overlap:, :] = sample_data
            # compute kaldi style feature
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                        frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            self.input_chunk[:, :self.chunk_overlap, :] = self.input_chunk[:, -self.chunk_overlap:, :].clone()
            self.input_chunk[:, self.chunk_overlap:, :] = xs.squeeze(0)

            # get vad status
            if self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "end" in speech_dict:
                    return_dict['status'] = 'el'
                    # reset state
                    self.vad_iterator.reset_states()
                else:
                    return_dict['status'] = 'cl'
            if not self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "start" in speech_dict:
                    return_dict['status'] = 'sl'
                    self.in_dialog = True
                    # self.vad_iterator.reset_states()
                else:  
                    # cache fbank feature
                    # << 1
                    self.history[:-1] = self.history[1:].clone()
                    # last history = input chunk
                    self.history[-1:] = self.input_chunk

            # return dict
            if return_dict['status'] == 'sl':
                # copy last chunk size chunks
                return_dict['feature_last_chunk'] = self.history[-self.last_chunk_size:].unsqueeze(1)
                return_dict['feature'] = self.input_chunk
                return_dict['history_feature'] = self.history
            elif return_dict['status'] == 'cl' or return_dict['status'] == 'el':
                return_dict['feature_last_chunk'] = None
                return_dict['feature'] = self.input_chunk
                return_dict['history_feature'] = self.history

        return return_dict
