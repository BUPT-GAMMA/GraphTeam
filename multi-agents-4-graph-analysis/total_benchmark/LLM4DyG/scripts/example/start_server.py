from argparse import ArgumentParser
import os
import json

"""Usage: Host Model
python start_server.py --model codellama2-13b -t run --device 0
"""

"""Usage: Clear Model
python start_server.py --model codellama2-13b -t clear --device 0
"""


parser = ArgumentParser()
parser.add_argument("--model", type=str, default="codellama2-13b")
parser.add_argument("-t", type=str, default="clear", choices="run clear".split())
parser.add_argument("--device", type=str, default="0")
args = parser.parse_args()

ROOT = "" # local model root path 

model = args.model
if model == "vicuna-7b":
    model_path = "vicuna-7b-v1.5"
    model_path = os.path.join(ROOT, model_path)
elif model == "vicuna-13b":
    model_path = "vicuna-13b-v1.5"
    model_path = os.path.join(ROOT, model_path)
elif model == "llama2-13b":
    model_path = "llama-2-13b"
    model_path = os.path.join(ROOT, model_path)
elif model == "llama2-7b":
    model_path = "NousResearch/Llama-2-7b-chat-hf"
elif model == "codellama2-13b":
    model_path = "codellama/CodeLlama-13b-Instruct-hf"
    # model_path = os.path.join(ROOT, model_path)
elif model == "chatglm2-6b":
    model_path = "chatglm2-6b"
    model_path = os.path.join(ROOT, model_path)
else:
    # or you can add your own model here
    raise NotImplementedError(f"{model} not implemented")

t = args.t
device = args.device

if t == "clear":
    os.system("pkill -f fastchat")
    os.system("rm ./*.log")
    os.system("rm ./*.log.*")
    
elif t == "run":
    num_gpus = len(device.split(','))
    os.system(f"CUDA_VISIBLE_DEVICES={device} python -m fastchat.serve.controller &")
    os.system(f"CUDA_VISIBLE_DEVICES={device} python -m fastchat.serve.model_worker --model-name '{model}' --model-path '{model_path}' --num-gpus {num_gpus} &")
    os.system(f"CUDA_VISIBLE_DEVICES={device} python -m fastchat.serve.openai_api_server &")
    with open("model.log", "w") as f:
        f.write(model)
    exit()
