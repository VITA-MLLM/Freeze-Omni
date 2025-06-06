from __future__ import print_function

import argparse
import os
import json
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins

import numpy as np

from copy import deepcopy
from threading import Timer
from flask import Flask, render_template, request
from flask_socketio import SocketIO, disconnect, emit

from web.parms import GlobalParams
from web.pool import TTSObjectPool, pipelineObjectPool
from web.pem import generate_self_signed_cert

def get_args():
    parser = argparse.ArgumentParser(description='Freeze-Omni')
    parser.add_argument('--model_path', required=True, help='model_path to load')
    parser.add_argument('--llm_path', required=True, help='llm_path to load')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--ip', required=True, help='ip of server')
    parser.add_argument('--port', required=True, help='port of server')
    parser.add_argument('--max_users', type=int, default=5)
    parser.add_argument('--llm_exec_nums', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument("--ngrok", action='store_true', help="use ngrok proxy")
    parser.add_argument("--ssl", action='store_true', help="use ssl")
    args = parser.parse_args()
    print(args)
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

"""
“模型作为服务器”策略来实现语音到语音对话系统。
首先，我们同时启动多个模型，并将它们视为服务器。
然后，当用户的VAD被触发时，语音将以块的形式发送到服务器，服务器将负责调度哪个空闲模型应该响应当前的块。
由于我们在推理过程中将语音编码器和LLM的所有kv-cache和CNN缓存分开，因此服务器只需要保存每个用户的推理缓存。
这样，服务器中的任何模型都可以响应任何用户的任何块，并且不需要指定哪个模型用作监视器或生成器。
"""
# init parms
configs = get_args()
# read server related config
server_configs = json.loads(open(configs.model_path + "/server.json").read())
# max users to connect
MAX_USERS = configs.max_users
# number of inference pipelines to use
PIPELINE_NUMS = configs.llm_exec_nums
# timeout to each user
TIMEOUT = configs.timeout

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init inference pipelines pool
pipeline_pool = pipelineObjectPool(size=PIPELINE_NUMS, configs=configs)
# inint speech decoder pool
tts_pool = TTSObjectPool(size=MAX_USERS, model_path=configs.model_path)

# init flask app
app = Flask(__name__, template_folder='../web/resources')
socketio = SocketIO(app)
# init connected users
connected_users = {}

def decoder(cur_hidden_state, cur_text, outputs, connected_users, sid, generate_num, last_text, is_last_chunk=False):
    """
    Decodes the current hidden state and text to generate audio segments using speech decoder.

    Parameters:
    - cur_hidden_state (list of torch.Tensor): The current hidden state of the language model.
    - cur_text (str): The current text to be synthesized.
    - connected_users (dict): A dictionary containing information about connected users.
    - sid (str): The session ID of the user.
    - is_last_chunk (bool, optional): Indicates if the current text is the last chunk of the input

    Returns:
    - int: The updated number of audio segments generated.
    """
    hidden_state_output = torch.cat(cur_hidden_state).squeeze(1)
    cur_text_procced = connected_users[sid][1].pipeline_obj.pipeline_proc.post_process(cur_text)
    print("Synthesis: ", [cur_text_procced])
    embeddings = connected_users[sid][1].pipeline_obj.pipeline_proc.model.llm_decoder.model.embed_tokens(
                    torch.tensor(connected_users[sid][1].pipeline_obj.pipeline_proc.model.tokenizer.encode(
                        cur_text_procced
                        )).cuda()
                    )
    codec_chunk_size = server_configs['decoder_first_chunk_size']
    codec_padding_size = server_configs['decoder_chunk_overlap_size']
    seg_threshold = server_configs['decoder_seg_threshold_first_pack']
    if generate_num != 0:
        codec_chunk_size = server_configs['decoder_chunk_size']
        seg_threshold = server_configs['decoder_seg_threshold']
    for seg in connected_users[sid][1].tts_obj.tts_proc.run(embeddings.reshape(-1, 896).unsqueeze(0), 
                                                            server_configs['decoder_top_k'], 
                                                            hidden_state_output.reshape(-1, 896).unsqueeze(0), 
                                                            codec_chunk_size, 
                                                            codec_padding_size,
                                                            server_configs['decoder_penalty_window_size'], 
                                                            server_configs['decoder_penalty'],
                                                            server_configs['decoder_N'], 
                                                            seg_threshold):
        if generate_num == 0:
            try:
                split_idx = torch.nonzero(seg.abs() > 0.03, as_tuple=True)[-1][0]
                seg = seg[:, :, split_idx:]
            except:
                print("Do not need to split")
                pass
        generate_num += 1
        if connected_users[sid][1].tts_over:
            connected_users[sid][1].tts_data.clear()
            connected_users[sid][1].whole_text = ""
            break
        connected_users[sid][1].tts_data.put((seg.squeeze().float().cpu().numpy() * 32768).astype(np.int16))
    return generate_num

def generate(outputs, sid):
    """
    Generates speech dialogue output based on the current state and user session ID.

    Parameters:
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.

    Returns:
    - None
    """
    # Stage3: start speak
    connected_users[sid][1].is_generate = True

    outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(None, **outputs)
    connected_users[sid][1].generate_outputs = deepcopy(outputs)

    cur_hidden_state = []
    cur_hidden_state.append(outputs['hidden_state'])

    connected_users[sid][1].whole_text = ""
    # Stage4: contiune speak until stat is set to 'sl'
    # use 'stop' to interrupt generation, stat need to be manually set as 'sl'  
    stop = False
    cur_text = ''
    last_text = ''
    generate_num = 0
    while True:
        if connected_users[sid][1].stop_generate:
            break
        if len(outputs['past_tokens']) > 500:
            stop = True
        if stop:
            break
        del outputs['text']
        del outputs['hidden_state']
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(None, **outputs)
        connected_users[sid][1].generate_outputs = deepcopy(outputs)
        if outputs['stat'] == 'cs':
            cur_hidden_state.append(outputs['hidden_state'])
            if "�" in outputs['text'][len(last_text):]:
                continue
            connected_users[sid][1].whole_text += outputs['text'][len(last_text):]
            cur_text += outputs['text'][len(last_text):]
            # print([connected_users[sid][1].whole_text])
            if generate_num == 0 or (len(cur_hidden_state) >= 20):
                suffix_list = [",", "，", "。", "：", "？", "！", ".", ":", "?","!", "\n"]
            else:
                suffix_list = ["。", "：", "？", "！", ".", "?","!", "\n"]
            if outputs['text'][len(last_text):].endswith(tuple(suffix_list)) and (len(cur_hidden_state) >= 4):
                if outputs['text'][len(last_text):].endswith(".") and last_text[-1].isdigit():
                    pass
                else:
                    if not connected_users[sid][1].tts_over:
                        if len(cur_hidden_state) > 0:
                            generate_num = decoder(cur_hidden_state, 
                                                   cur_text, outputs, 
                                                   connected_users, 
                                                   sid, 
                                                   generate_num, 
                                                   last_text)
                            cur_text = ""
                            cur_hidden_state = []
            last_text = outputs['text']
        else:
            break
    if not connected_users[sid][1].tts_over:
        if len(cur_hidden_state) != 0:
            generate_num = decoder(cur_hidden_state, 
                                   cur_text, outputs, 
                                   connected_users, 
                                   sid, 
                                   generate_num, 
                                   last_text, 
                                   is_last_chunk=True)
            cur_text = ""
    connected_users[sid][1].is_generate = False

def llm_prefill(data, outputs, sid, is_first_pack=False):
    """
    Prefills the LLM of speech dialogue system using speech.

    Parameters:
    - data (dict): A dictionary containing the current state of the user's input, 
                   including features and status.
    - outputs (dict): A dictionary containing the current state of the dialogue system.
    - sid (str): The session ID of the user.
    - is_first_pack (bool, optional): Indicates if the current input packet is the first one in a new conversation
    """

    if data['status'] == 'sl':
        # Satge1: start listen
        # stat will be auto set to 'cl' after Stage1
        outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
                                                                     data['feature'], 
                                                                     **outputs)
        print(f"sl -> speech_dialogue outputs stat: {outputs['stat']}")
    
    if data['status'] == 'el':
        connected_users[sid][1].wakeup_and_vad.in_dialog = False
        print("Sid: ", sid, " Detect vad time out")

    if data['status'] == 'cl':
        if outputs['stat'] == 'cl':
            # Stage2: continue listen
            # stat will be auto set to 'ss' when endpoint is detected
            outputs = connected_users[sid][1].pipeline_obj.pipeline_proc.speech_dialogue(
                                                                         data['feature'], 
                                                                         **outputs)
            print(f"cl -> speech_dialogue outputs stat: {outputs['stat']}")
        if is_first_pack:
            outputs['stat'] = 'cl'
        if outputs['stat'] == 'el':
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            print("Sid: ", sid, " Detect invalid break")
        if outputs['stat'] == 'ss':
            connected_users[sid][1].interrupt()
            print("Sid: ", sid, " Detect break")
            connected_users[sid][1].wakeup_and_vad.in_dialog = False
            generate_thread = threading.Thread(target=generate, args=(deepcopy(outputs), sid))
            generate_thread.start()
    return outputs

def send_pcm(sid):
    """
    Sends PCM audio data to the dialogue system for processing.

    Parameters:
    - sid (str): The session ID of the user.
    """

    chunk_size = connected_users[sid][1].wakeup_and_vad.get_chunk_size()

    print("Sid: ", sid, " Start listening")
    while True:
        if connected_users[sid][1].stop_pcm:
            print("Sid: ", sid, " Stop pcm")
            connected_users[sid][1].stop_generate = True
            connected_users[sid][1].stop_tts = True
            break
        time.sleep(0.01)
        e = connected_users[sid][1].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue
        print("Sid: ", sid, " Received PCM data: ", len(e))
        res = connected_users[sid][1].wakeup_and_vad.predict(np.float32(e))
        print(f"wakeup_and_vad.predict -> res status: {res['status']}")
        
        if res['status'] == 'sl':
            print("Sid: ", sid, " Vad start")
            outputs = deepcopy(connected_users[sid][1].generate_outputs)
            outputs['adapter_cache'] = None
            outputs['encoder_cache'] = None
            outputs['pe_index'] = 0
            outputs['stat'] = 'sl'
            outputs['last_id'] = None
            if 'text' in outputs:
                del outputs['text']
            if 'hidden_state' in outputs:
                del outputs['hidden_state']

            send_dict = {}
            for i in range(len(res['feature_last_chunk'])):
                if i == 0:
                    send_dict['status'] = 'sl'
                else:
                    send_dict['status'] = 'cl'
                send_dict['feature'] = res['feature_last_chunk'][i]
                outputs = llm_prefill(send_dict, outputs, sid, is_first_pack=True)
            send_dict['status'] = 'cl'
            send_dict['feature'] = res['feature']
            outputs = llm_prefill(send_dict, outputs, sid)

        elif res['status'] == 'cl' or res['status'] == 'el':
            send_dict = {}
            send_dict['status'] = res['status']
            send_dict['feature'] = res['feature']
            outputs = llm_prefill(send_dict, outputs, sid)

def disconnect_user(sid):
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to time out")
        socketio.emit('out_time', to=sid) 
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]

@app.route('/')
def index():
    return render_template('demo.html')

@socketio.on('connect')
def handle_connect():
    if len(connected_users) >= MAX_USERS:
        print('Too many users connected, disconnecting new user')
        emit('too_many_users')
        return

    sid = request.sid
    connected_users[sid] = []
    connected_users[sid].append(Timer(TIMEOUT, disconnect_user, [sid]))
    connected_users[sid].append(GlobalParams(tts_pool, pipeline_pool))
    connected_users[sid][0].start()
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
    tts_pool.print_info()
    pipeline_pool.print_info()
    print(f'User {sid} connected')

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][1].interrupt()
        connected_users[sid][1].stop_pcm = True
        connected_users[sid][1].release()
        time.sleep(3)
        del connected_users[sid]
    tts_pool.print_info()
    pipeline_pool.print_info()
    print(f'User {sid} disconnected')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][0].cancel()
        connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid][0].start()
        connected_users[sid][1].interrupt()
        socketio.emit('stop_tts', to=sid)
        connected_users[sid][1].reset()
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('prompt_text')
def handle_prompt_text(text):
    sid = request.sid
    if sid in connected_users:
        connected_users[sid][1].set_prompt(text)
        print("Sid: ", sid, "Prompt set as: ", text)
        socketio.emit('prompt_success', to=sid)
    else:
        disconnect()

@socketio.on('audio')
def handle_audio(data):
    sid = request.sid
    if sid in connected_users:
        if not connected_users[sid][1].tts_data.is_empty():
            connected_users[sid][0].cancel()
            connected_users[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            connected_users[sid][0].start()
            output_data = connected_users[sid][1].tts_data.get()
            if output_data is not None:
                print("Sid: ", sid, "Send TTS data")
                emit('audio', output_data.astype(np.int16).tobytes())

        if connected_users[sid][1].tts_over_time > 0:
            socketio.emit('stop_tts', to=sid)
            connected_users[sid][1].tts_over_time = 0
        
        data = json.loads(data)
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        sample_rate = data['sample_rate']
        
        connected_users[sid][1].pcm_fifo_queue.put(audio_data.astype(np.float32) / 32768.0)

    else:
        disconnect()

def ngrok_proxy(port):
    """
    run `ngrok config add-authtoken $NGROK_TOKEN`
    """
    from pyngrok import ngrok
    import nest_asyncio

    ngrok_tunnel = ngrok.connect(port)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()


if __name__ == "__main__":
    print("Start Freeze-Omni sever") 
    if configs.ssl:
        cert_file = "web/resources/cert.pem"
        key_file = "web/resources/key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            generate_self_signed_cert(cert_file, key_file)

    if configs.ngrok and not configs.ssl:
        ngrok_proxy(configs.port)

    if configs.ssl:
        socketio.run(app, host=configs.ip, port=configs.port, ssl_context=(cert_file, key_file))
    else:
        socketio.run(app, host=configs.ip, port=configs.port)
