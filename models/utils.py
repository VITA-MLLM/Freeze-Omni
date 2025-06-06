import re
import os

import yaml
import torch
import numpy as np

from models.audioLLM import AudioLLM
from models.encoder.cmvn import GlobalCMVN, load_cmvn
from models.encoder.encoder import speechEncoder

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        print('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        print('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    
    # load parm from checkpoint
    model.load_state_dict(checkpoint, strict=False)

    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    # get configs
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.safe_load(fin)
    return configs

def init_encoder_llm(configs):
    """
    init Modeling of speech input (encoder and audio llm)
    """
    if configs['cmvn_file'] is not None:
        # read cmvn
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        # init cmvn layer
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    # init speech encoder (几个下采样卷积层和几个 Transformer)
    # 块式流式语音编码器将输入语音特征转换为高维表示
    encoder = speechEncoder(input_dim, global_cmvn=global_cmvn, **configs['encoder_conf'])
    # init audioLLM 
    # 默认： 适配器:CNN 仅包含几个下采样卷积层, 
    # 适配器模块将高维表示映射到主干LLM的嵌入空间中
    # 使用下采样的原因是为了降低语音特征的帧率，提高预填充阶段LLM的速度，降低延迟
    model = AudioLLM(encoder=encoder, **configs['model_conf'])

    return model

def print_outputs(outputs: dict):
    print_str = ""
    for key, item in outputs.items():
        if isinstance(item, (torch.Tensor, np.ndarray)):
            print_str += f"{key} shape:{item.shape} "
        if isinstance(item, (str, int)):
            print_str += f"{key}:{item} "
        if isinstance(item, (list, tuple)):
            print_str += f"{key} len:{len(item)} "
    return print_str