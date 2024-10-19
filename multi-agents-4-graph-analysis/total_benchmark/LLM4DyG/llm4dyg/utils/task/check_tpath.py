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
    
def select(context, path_length, label, num_nodes):
    path = np.random.choice(num_nodes, path_length, replace = False)
    judge = judge_path(context, path)
    if label == judge:
        return path
    else:
        return []

def select_try(context, path_length, label, num_nodes,cnt = 100):
    for i in range(cnt):
        path = select(context, path_length, label, num_nodes)
        if len(path):
            return path
    assert False, "no answer"
    
def generate_path(context, path_length, label, num_nodes):
    iters = [x for x in permutations(np.arange(num_nodes), path_length)]
    np.random.shuffle(iters)
    for path in iters:
        judge = judge_path(context, path)
        if label == judge:
            return path
    assert False, "no answer"
    
class DyGraphTaskCheckTPath(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        # select num_nodes
        nodes = list(set(list(context[:, :2].flatten())))
        num_nodes = len(nodes)
        # path_length = np.random.choice(np.arange(3, 1 + min(num_nodes, 5)))
        path_length = 3
        assert num_nodes >= path_length 
        label = kwargs['label']
        answer = 'yes' if label else 'no'
        
        # answer 
        path = generate_path(context, path_length, label, num_nodes)
        path = list(map(int,path))
        query = path

        context = context.tolist()
        qa = {
            "context": context,
            "query": query,
            "answer": answer,
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer whether a path is chronological in the dynamic graph. The time of the edges in a chronological path from source node to target node must not decrease, e.g., [2, 3, 5] is a chronological path in the dynamic graph [(2, 3, 0), (3, 5, 1)]\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as yes or no at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 2, 0), (2, 3, 1), (1, 2, 2), (3, 1, 3)],
                [0, 2 ,3],
                'yes'
             ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"Is the path {query} a chronological path?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"Answer:\s*(yes|no|Yes|No)\s*", response)
        if match:
            s = match.group(1).lower()
            metric = (s == ans)
            return metric
        else:
            match = re.search(r"(yes|no|Yes|No)", response)
            if match:
                s = match.group(1).lower()
                metric = (s == ans)
                return metric
            else:
                return -1