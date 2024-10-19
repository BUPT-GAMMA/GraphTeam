from .num_time import DyGraphTaskNumTime
from .what_node import DyGraphTaskWhatNode
from .when_link import DyGraphTaskWhenLink
from .num_edge import DyGraphTaskNumEdge
from .which_neighbor import DyGraphTaskWhichNeighbor
from .check_tpath import DyGraphTaskCheckTPath
from .find_tpath import DyGraphTaskFindTPath
from .check_tclosure import DyGraphTaskCheckTClosure
from .when_tclosure import DyGraphTaskWhenTClosure
from .sort_edge import DyGraphTaskSortEdge
from .when_connect import DyGraphTaskWhenConnect
from .link_pred import DyGraphTaskLinkPred

def load_task(task, args):
    if task == "num_time":
        return DyGraphTaskNumTime(task, args)
    elif task == "when_link":
        return DyGraphTaskWhenLink(task, args)
    elif task == "what_node":
        return DyGraphTaskWhatNode(task, args)
    elif task == "num_edge":
        return DyGraphTaskNumEdge(task, args)
    elif task == "which_neighbor":
        return DyGraphTaskWhichNeighbor(task, args)
    elif task == "check_tpath":
        return DyGraphTaskCheckTPath(task, args)
    elif task == "find_tpath":
        return DyGraphTaskFindTPath(task, args)
    elif task == "check_tclosure":
        return DyGraphTaskCheckTClosure(task, args)
    elif task == "when_tclosure":
        return DyGraphTaskWhenTClosure(task, args)
    elif task == "sort_edge":
        return DyGraphTaskSortEdge(task, args)
    elif task == "when_connect":
        return DyGraphTaskWhenConnect(task, args)
    elif task == "link_pred":
        return DyGraphTaskLinkPred(task, args)
    else:
        raise NotImplementedError(f"{task} not implemented")
    
import math
import json  
from llm4dyg.utils.task.find_tpath import judge_path
from itertools import permutations, combinations
import numpy as np
import os
from tqdm import tqdm
def get_random_base(task, T, N, get_task_folder):
    """
    Calculate the random base for a given task.

    Args:
        task (str): The task for which to calculate the random base.
        T (int): The total number of elements for the task.
        N (int): The total number of nodes for the task.
        get_task_folder (function): to get the root folder of the task 

    Returns:
        float: The calculated random base for the given task.

    Raises:
        ValueError: If the task is not recognized.

    """
    
    if task == 'when_link':
        combinations = sum(math.comb(T, i) for i in range(1, T+1))
        num_solution = combinations
        return 1/num_solution
    if task == 'when_connect':
        num_solution = T
        return 1/num_solution
    if task == 'when_tclosure':
        num_solution = T
        return 1/num_solution
    if task == 'what_node':
        num_solution = sum(math.comb(N, i) for i in range(1, N+1))
        return 1/num_solution
    if task == 'which_neighbor':
        num_solution = sum(math.comb(N, i) for i in range(1, N+1))
        return 1/num_solution
    if task == 'check_tclosure':
        num_solution = 2
        return 1/num_solution
    if task == 'check_tpath':
        num_solution = 2
        return 1/num_solution
    if task == 'find_tpath':
        def generate_solutions(context, num_nodes, n1):
            l1 = np.arange(num_nodes)
            l1 = l1[l1!= n1]
            
            iters = [[n1] + list(x)  for x in permutations(l1, 2)]
            num = 0
            for path in iters:
                judge = judge_path(context, path)
                num += judge 
            return num
        def get_base(task_folder, folder_name):
            folder_path = os.path.join(task_folder, folder_name)
            file_path = os.path.join(folder_path, "qa.json")
            graph_path = os.path.join(folder_path, f"graph.json")
            
            graph = json.load(open(graph_path, "r"))
            qa= json.load(open(file_path, "r"))
            T, N, p = graph['T'], graph['N'], graph['p']
            num_space = math.perm(N, 2)
            num_solution = generate_solutions(qa['context'], N, qa['query'][0])
            return num_solution/num_space, N
        
        
        task_folder = get_task_folder(task)
        files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
        bases = []
        for folder_name in tqdm(files):
            base = get_base(task_folder, folder_name)
            if base[-1] == N:
                bases.append(base[0])
        bases = np.array(bases).mean()
        return bases
    
    if task == 'sort_edge':
        def get_base(task_folder, folder_name):
            folder_path = os.path.join(task_folder, folder_name)
            file_path = os.path.join(folder_path, "qa.json")
            graph_path = os.path.join(folder_path, f"graph.json")
            
            graph = json.load(open(graph_path, "r"))
            qa= json.load(open(file_path, "r"))
            T, N, p = graph['T'], graph['N'], graph['p']
            context = np.array(graph['edge_index'])
            num_edge = len(context)
            ts = list(set(list(context[:, 2])))
            num_space = math.perm(num_edge, num_edge)
            num_solution = 0
            for t in ts:
                num_solution += math.perm(len(context[context[:, 2] == t]), len(context[context[:, 2] == t]))
            return num_solution/num_space, N
        
        
        task_folder = get_task_folder(task)
        files = json.load(open(os.path.join(task_folder, "prompt_files.json"), "r"))["files"]
        bases = []
        for folder_name in tqdm(files):
            base = get_base(task_folder, folder_name)
            if base[-1] == N:
                bases.append(base[0])
        bases = np.array(bases).mean()
        return bases