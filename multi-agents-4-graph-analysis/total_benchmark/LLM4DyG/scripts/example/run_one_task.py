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

# RUN
MRun = Runner
runner = MRun(args, try_all = True)
runner.execute(log_dir)