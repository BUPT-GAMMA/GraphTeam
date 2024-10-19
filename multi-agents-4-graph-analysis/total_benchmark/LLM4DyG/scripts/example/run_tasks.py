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

tasks = "when_link when_connect when_tclosure what_node which_neighbor check_tclosure check_tpath find_tpath sort_edge".split()
task_folder = args.task_folder = get_task_folder(args.task)

# define runner
class MRun(Runner):
    def show(self,dir):
        args = self.args
        table = []
        for task in tasks:
            task_folder = get_task_folder(task)
            if not os.path.exists(task_folder):
                continue
            model = args.model
            obj_task = load_task(task, args)
            files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
            for folder_name in tqdm(files):
                folder_path = os.path.join(task_folder, folder_name)
                file_path = os.path.join(folder_path, "qa.json")
                answer_path = os.path.join(folder_path, f"answer_{model}.json")
                graph_path = os.path.join(folder_path, f"graph.json")
                
                qa = json.load(open(file_path, "r"))
                answer = json.load(open(answer_path, "r"))
                graph = json.load(open(graph_path, "r"))
                T, N, p = graph['T'], graph['N'], graph['p']
                metric = obj_task.evaluate(qa, answer["content"])
                table.append([task, metric, T, N, p])
        df = pd.DataFrame(table, columns= "task m T N p".split())
        # print(df)
       
        # fail 
        print('#'*10,'failing rate')       
        NS = sorted(list(set(list(df['N'].values))))
        accs = []
        for task in tasks:
            for N in NS:
                acc = df.query(f'task == "{task}" and N == {N} ')['m'].to_numpy()
                fail = len(acc[acc<0])/len(acc)
                acc = fail
                accs.append(acc)
            acc = df.query(f'task == "{task}"')['m'].to_numpy()
            fail = len(acc[acc<0])/len(acc)
            acc = fail
            accs.append(acc)
        accs = np.array(accs)
        accs = accs.reshape(len(tasks),len(NS)+1).T
        df2 = pd.DataFrame(accs, columns = tasks, index = NS + ['Avg'])
        print("#"*10,"fail rate")
        print(df2.applymap(lambda x: round(x*100,2)))
        
        # table
        print('#'*10,'accuracy')       
        NS = sorted(list(set(list(df['N'].values))))
        accs = []
        for task in tasks:
            for N in NS:
                acc = df.query(f'task == "{task}" and N == {N}')['m'].to_numpy()
                acc[acc<0] = 0
                acc = acc.mean()
                accs.append(acc)
            acc = df.query(f'task == "{task}" ')['m'].to_numpy()
            acc[acc<0] = 0
            acc = acc.mean()
            accs.append(acc)
        accs = np.array(accs)
        accs = accs.reshape(len(tasks),len(NS)+1).T
        df2 = pd.DataFrame(accs, columns = tasks, index = NS + ['Avg'])
        print(df2.applymap(lambda x: round(x*100,1)))
        
        # get base
        from llm4dyg.utils import get_random_base
        accs = []
        for task in tasks:
            for N in NS:
                acc = get_random_base(task, T, N, get_task_folder)
                accs.append(acc)
        accs = np.array(accs)
        accs = accs.reshape(len(tasks),len(NS)).T
        accs = np.concatenate([accs, accs.mean(axis = 0, keepdims = True)], axis = 0)
        df3 = pd.DataFrame(accs, columns = tasks, index = NS + ['Avg'])
        print(df3.applymap(lambda x: round(x*100,1)))
        
        df4 = df2- df3
        df4 = df4.applymap(lambda x: f"+{round(x*100,1)}" if x>=0 else f"-{round(-x*100,1)}")
        
        df2['model'] = "model"
        df3['model'] = "Random"
        df4['model'] = "\Delta"
        
        df5 = pd.concat([df2, df3, df4])
        df5 = df5.applymap(lambda x: round(x*100,1) if isinstance(x, float) else x)
        df5 = df5.reset_index()
        df5 = df5.sort_values(by = 'index model'.split())
        print('#'*10,'accuracy')
        print(df5)
    
for task in tasks:
    args.task = task
    args.task_folder = get_task_folder(args.task)
    runner = MRun(args, try_all = True)
    runner.execute(log_dir)
    
    