import os
from llm4dyg.utils import  remove_dir, load_task
import json
from tqdm import trange
from config import get_args
import llm4dyg
from libwon.utils import setup_seed
from llm4dyg.utils import send_prompt, DyGraphPrompt, DyGraphGenERCon
from tqdm import tqdm
import random
from llm4dyg.runner import Runner
import numpy as np
import pandas as pd
import time

# set args
args = get_args()
log_dir = args.log_dir
file_name = os.path.splitext(os.path.split(__file__)[-1])[0]
def get_task_folder(task):
    return os.path.join(log_dir, f"{file_name}", f"{task}")
task_folder = args.task_folder = get_task_folder(args.task)
T, N, p, seed = args.T[0], args.N[0], args.p[0], 0
folder_setting = f"{args.T}_{N}_{p}_{seed}"
task = args.task

# generate data
dygen = DyGraphGenERCon()                                               # data generator
obj_task = load_task(task, args)                                        # task 
dygprompt = DyGraphPrompt(obj_task, args = args)                        # prompt generator
info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)    # generate data
qa = obj_task.generate_qa(info)                                         # generate qa       
prompt_qa = dygprompt.generate_prompt_qa(**qa)                          # generate prompt qa
print('#'*10,'prompt_qa:\n', prompt_qa)

# generate response
model = args.model
prompt = prompt_qa['prompt']
answer = send_prompt(model, prompt, temperature = args.temperature, max_tokens = args.max_tokens)
print('#'*10,'answer:\n', answer)

# score
metric = obj_task.evaluate(qa, answer["content"])
print('#'*10,'metric:\n', metric)




