from .base import DyGraphTask
import numpy as np
import re
from itertools import permutations

def find_edge_t(context, edge, curt): # find earlest t>= curt
    n1, n2 = edge
    for e1, e2, t in context:
        if (e1 == n1 and e2 == n2) or (e1 == n2 and e2 == n1) and t>=curt:
            return t
    return -1

def judge_path(context, path):
    ts = []
    curt = 0
    for i in range(len(path)-1):
        n1, n2 = path[i], path[i+1]
        curt = find_edge_t(context, (n1,n2), curt)
        if curt == -1: # go back time
            return False
    return True
    
def generate_start_node(context, num_nodes):
    iters = [x for x in permutations(np.arange(num_nodes), 3)]
    np.random.shuffle(iters)
    for path in iters:
        judge = judge_path(context, path)
        if judge:
            return int(path[0])
    assert False, "no answer"
    
class DyGraphTaskFindTPath(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        # select num_nodes
        nodes = list(set(list(context[:, :2].flatten())))
        num_nodes = len(nodes)
        start_node = generate_start_node(context, num_nodes)
        
        # answer 
        context = context.tolist()
        qa = {
            "context": context,
            "query": [start_node],
            "answer": [],
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer whether a path is chronological in the dynamic graph. The time of the edges in a chronological path from source node to target node must not decrease, e.g., [2, 3, 5] is a chronological path in the dynamic graph [(2, 3, 0), (3, 5, 1)]\n"
    
    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 2, 0), (2, 3, 1), (1, 2, 2), (3, 1, 3)],
                [0],
                [0, 2, 3]
             ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"Find a chronological path starting at node {query[0]} with a length no less that 3. \n"
    
    def evaluate(self, qa, response):
        context = qa['context']
        start_node = qa['query'][0]
        match = re.search(r"Answer:\s*\[([\d,\s]+)\]", response)
        if match:
            numbers_str = match.group(1)
            path = [int(num) for num in numbers_str.split(',')]
            if path[0] != start_node:
                return False
            if len(path) < 3:
                return False
            metric = judge_path(context, path)
            return metric
        
        match = re.search(r"""[O|o]utput.*\s*[:`'"]*\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            path = [int(num) for num in numbers_str.split(',')]
            if path[0] != start_node:
                return False
            if len(path) < 3:
                return False
            metric = judge_path(context, path)
            return metric
        
        match = re.search(r"""[c|C]hronological path.*is.*\s*[:`'"]*\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            path = [int(num) for num in numbers_str.split(',')]
            if path[0] != start_node:
                return False
            if len(path) < 3:
                return False
            metric = judge_path(context, path)
            return metric
        
        match = re.search(r"""[A|a]nswer.*is.*\s*[:`'"]*\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            path = [int(num) for num in numbers_str.split(',')]
            if path[0] != start_node:
                return False
            if len(path) < 3:
                return False
            metric = judge_path(context, path)
            return metric
        return -1
