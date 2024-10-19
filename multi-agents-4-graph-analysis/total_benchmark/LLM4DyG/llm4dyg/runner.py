import os
from .utils import  remove_dir, load_task
import json
from .utils import send_prompt, DyGraphPrompt, DyGraphGenERCon
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from .utils.misc import TPMController

class Runner:
    def __init__(self, args, try_all = False) -> None:
        """
        A class that provides methods for running tasks and evaluating results.

        Args:
            args (object): An object containing the arguments for the Runner.
            try_all (bool, optional): Whether to try running all tasks continuously. Defaults to False.

        Attributes:
            args (object): An object containing the arguments for the Runner.
            try_all (bool): Whether to try running all tasks continuously.

        """
        self.args = args
        self.try_all = try_all
        
    def check(self, task_folder):
        """
        Check the status of tasks in a given folder.

        Args:
            task_folder (str): The path to the folder containing the tasks.

        Returns:
            int: The number of tasks that need to be run.

        """
        args = self.args
        model = args.model
        files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
        finish = []
        torun = []
        sdict = {"num_edges":[], "num_nodes":[], "num_time":[]}
        for i, folder_name in enumerate(files):
            folder_path = os.path.join(task_folder, folder_name)
            graph = json.load(open(os.path.join(folder_path, "graph.json")))
            for k, v in sdict.items():
                v.append(graph[k])
            answer_path = os.path.join(folder_path, f"answer_{model}.json")
            if os.path.exists(answer_path):
                finish.append(i)
            else:
                torun.append(i)
        print(f"Finish {len(finish)}, ToRun {len(torun)}")
        print("".join(f"{k}:{np.mean(v):.2f}+-{np.std(v):.2f} \t" for k,v in sdict.items()))
        return len(torun)
        
    
    def generate_save(self, dir, T, N, p, seed, *targs ,**kwargs):
        """
        Generates and saves dynamic graph data, QA data, and prompt-QA data.

        Args:
            dir (str): The directory where the data will be saved.
            T (int): The time steps for the dynamic graph.
            N (int): The number of nodes in the dynamic graph.
            p (float): The probability of an edge being present in the dynamic graph.
            seed (int): The seed for random number generation.
            task (str): The task for which to generate the data.
            *targs: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The folder setting where the data is saved.
        """
        
        folder_setting = f"{T}_{N}_{p}_{seed}"
        args = self.args
        task = args.task
        
        # init
        dygen = DyGraphGenERCon()
        obj_task = load_task(task, args)
        dygprompt = DyGraphPrompt(obj_task, args = args)
        
        # generate prompt_qa
        info = dygen.sample_dynamic_graph(T = T, N = N , p = p, seed = seed)
        qa = obj_task.generate_qa(info, *targs, **kwargs)
        prompt_qa = dygprompt.generate_prompt_qa(**qa)

        # file paths
        folder = os.path.join(dir, folder_setting)
        os.makedirs(folder, exist_ok=True)
        info_file = os.path.join(folder, f"graph.json")
        qa_file = os.path.join(folder, f"qa.json")
        prompt_qa_file = os.path.join(folder, f"prompt_qa.json")
        
        # write files
        json.dump(info, open(info_file, "w"))
        json.dump(qa, open(qa_file, "w"))
        json.dump(prompt_qa, open(prompt_qa_file, "w"), indent=4)
        return folder_setting

    # run
    def gen(self, dir):
        """
        Generate prompt files based on the given directory.

        Args:
            dir (str): The directory to save the generated prompt files.

        Returns:
            None
        """
        print('generate prompt files for task', self.args.task, 'in', self.args.task_folder)
        args = self.args
        os.makedirs(dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(dir, 'args.json'), "w"), indent = 4)
        prompt_files = []
        label = 0   
        task = args.task
        for T in args.T:
            for N in args.N:
                for p in args.p:
                    seed = 0
                    pf_set = []
                    while len(pf_set) < args.num_seed:
                        # if True:
                        try:
                            folder_setting = self.generate_save(dir, T, N, p, seed, label = label)
                            pf_set.append(folder_setting)
                            label = not label
                        except Exception as e:
                            print(e)
                        seed +=1
                    prompt_files.extend(pf_set)
                    
        json.dump({"files": prompt_files}, open(os.path.join(dir, f"prompt_files.json"), "w"))
    
    def run_one(self, task_folder):
        args = self.args
        model = args.model
        con = TPMController(start_token = args.start_token)
        files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
        with tqdm(files) as bar:
            for folder_name in bar:
                try:
                    folder_path = os.path.join(task_folder, folder_name)
                    file_path = os.path.join(folder_path, "prompt_qa.json")
                    answer_path = os.path.join(folder_path, f"answer_{model}.json")
                    prompt = json.load(open(file_path, "r"))['prompt']
                    if os.path.exists(answer_path):
                        continue
                    token = con.get_token()
                    bar.set_postfix(token = token)
                    answer = send_prompt(model, prompt, temperature = args.temperature, max_tokens = args.max_tokens)
                    con.time_token()
                    con.use_token(answer['total_tokens'])
                    json.dump(answer, open(answer_path, "w"))
                except Exception as e:
                    print(e)
                    
    def run(self, task_folder):
        print('get answers for task', self.args.task, 'in', self.args.task_folder)
        if self.try_all:
            while 1:
                self.run_one(task_folder)
                torun = self.check(task_folder)
                if torun == 0:
                    break
                print("Continue Try to Run")
                time.sleep(5)
        else:
            self.run_one(task_folder)

    def evaluate(self, task_folder):
        args = self.args
        model = args.model
        task = args.task
        obj_task = load_task(task, args)
        files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
        metrics = []
        # total_tokens = []
        # prompt_tokens = []
        # completion_tokens = []
        wrong_folders = []
        fail_folders = []
        num_times = []
        num_edges = []
        num_nodes = []
        for folder_name in tqdm(files):
            folder_path = os.path.join(task_folder, folder_name)
            file_path = os.path.join(folder_path, "qa.json")
            answer_path = os.path.join(folder_path, f"answer_{model}.json")
            graph_path = os.path.join(folder_path, f"graph.json")
            
            qa = json.load(open(file_path, "r"))
            answer = json.load(open(answer_path, "r"))
            graph = json.load(open(graph_path, "r"))
            
            metric = obj_task.evaluate(qa, answer["content"])
            metrics.append(metric)
            
            if metric< 0: 
                print(f"Fail: {folder_name}")
                fail_folders.append(folder_name)
            if metric == 0:
                print(f"Error: {folder_name}")
                wrong_folders.append(folder_name)
                
            # total_tokens.append(answer['total_tokens'])
            # prompt_tokens.append(answer['prompt_tokens'])
            # completion_tokens.append(answer["completion_tokens"])
            num_times.append(graph['num_time'])
            num_edges.append(graph['num_edges'])
            num_nodes.append(graph['num_nodes'])

        num_fail = len([m for m in metrics if m<0 ])
        num_all = len(metrics)
        average_acc = sum([m for m in metrics if m>=0])/num_all
        fail_rate = num_fail / num_all
        # total_tokens = sum(total_tokens)
        # average_tokens = total_tokens / num_all

        results = {
            "fail_rate": fail_rate,
            "average_acc": average_acc,
            # "average_tokens": average_tokens,
            # "total_tokens": total_tokens,
            "average_num_times": sum(num_times)/num_all,
            "average_num_edges": sum(num_edges)/num_all,
            "average_num_nodes": sum(num_nodes)/num_all,
            "metrics": metrics,
            # "total_tokens": total_tokens,
            # "prompt_tokens": prompt_tokens,
            # "completion_tokens": completion_tokens,
            "wrong_folders": wrong_folders,
            "fail_folders": fail_folders,
        }

        json.dump(results, open(os.path.join(task_folder, f"results_{model}.json"), "w"), indent=4)
        print(f"Task: {task}, Model: {model}")
        # print(f"Fail Rate: {fail_rate:.2f}, Average Acc: {average_acc:.4f}, Average Tokens: {average_tokens:.2f}, Total Tokens: {total_tokens}")
        print(f"Fail Rate: {fail_rate:.2f}, Average Acc: {average_acc:.4f}")
        print(f"Num_time : {np.mean(num_times):.2f}+-{np.std(num_times):.2f} Num_edges : {np.mean(num_edges):.2f}+-{np.std(num_edges):.2f} Num Nodes : {np.mean(num_nodes):.2f}+-{np.std(num_nodes):.2f}")

        
    def show(self, dir):
        args = self.args
        
        table = []
        task = args.task
        task_folder = args.task_folder
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
            # metric = max(metric, 0)
            table.append([task, metric, T, N, p])
        df = pd.DataFrame(table, columns= "task m T N p".split())
        print(df)
        
        
        TS = sorted(list(set(list(df['T'].values))))
        NS = sorted(list(set(list(df['N'].values))))
        PS = sorted(list(set(list(df['p'].values))))
        
        for p in PS:
            accs = []
            for T in TS:
                for N in NS:
                    acc = df.query(f'task == "{task}" and T == {T} and N == {N} and p == {p} ')['m'].to_numpy()
                    acc[acc<0] = 0
                    acc = acc.mean()
                    accs.append(acc)
            accs = np.array(accs)
            accs = accs.reshape(len(TS),len(NS))
            df2 = pd.DataFrame(accs, columns = NS, index = TS)
            print('task:', task, ' density:', p)
            print( df2)
        
    def execute(self, dir):
        args = self.args
        task_folder = args.task_folder
        if args.t == "clear":
            remove_dir(task_folder)
        elif args.t == "gen":
            self.gen(task_folder)
        elif args.t == "run":
            self.run(task_folder)
        elif args.t == "eval":
            self.evaluate(task_folder)
        elif args.t == "check":
            self.check(task_folder)
        elif args.t == "show":
            self.show(args.log_dir)
        else:
            raise NotImplementedError

