from .base import DyGraphTask
import numpy as np
import re

def find_neighbors(x, con):
    res = set()
    for e1,e2,t in con:
        if e1 == x:
            res.add(e2)
        if e2 == x:
            res.add(e1)
    return list(res)
    

def cands(n1, context, t):
    nodes_not_after = set(find_neighbors(n1,context[context[:, 2] <= t]))
    nodes_after = set(find_neighbors(n1,context[context[:, 2] > t]))
    nodes_after.difference_update(nodes_not_after) # only after time t
    nodes_after = list(map(int,nodes_after))
    return nodes_after

def select(context):
    # select time
    times = list(set(context[:,2].tolist()))
    t_max = np.max(times)
    t_min = np.min(times)
    t = int(np.random.choice(np.arange(t_min + 1, t_max + 1)))
    
    # select n1
    nodes = list(set(list(context[:, :2].flatten())))
    n1 = int(np.random.choice(nodes))

    # select candidates
    nodes_after = cands(n1, context, t)
    return t, n1, nodes_after

def select_try(context, cnt = 100):
    for i in range(cnt):
        t, n1, nodes_after = select(context)
        if nodes_after:
            break
    if not nodes_after:
        assert False, "no answer"
    return t, n1, nodes_after
        
class DyGraphTaskWhichNeighbor(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        t, n1, nodes_after = select_try(context)
                
        query = [t, n1]

        answer = nodes_after
        
        context = context.tolist()
        qa = {
            "context": context,
            "query": query,
            "answer": answer,
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer what nodes are linked with one node only after some time in the dynamic graph.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(1, 2, 1), (0, 3, 2), (1, 2, 5), (3, 1, 6)],
                [5, 1], 
                [3]
             ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"What nodes are linked with node {query[1]} after or at time {query[0]} but not linked before time {query[0]}?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        try:
            match = re.search(r"Answer:\s*\[([\d,\s]+)\]", response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num) for num in numbers_str.split(',')]
                metric = (set(numbers) == set(ans))
                return metric
            
            match = re.search(r"""answer is\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num) for num in numbers_str.split(',')]
                metric = (set(numbers) == set(ans))
                return metric
            
            match = re.search(r"""time [\d]+ are\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num) for num in numbers_str.split(',')]
                metric = (set(numbers) == set(ans))
                return metric
            
            match = re.search(r"""time [\d]+ is\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num) for num in numbers_str.split(',')]
                metric = (set(numbers) == set(ans))
                return metric
            
            match = re.search(r"""[O|o]utput.*\s*[:`'"]*\s*\[([\d,\s]+)\]""", response)
            if match:
                numbers_str = match.group(1)
                numbers = [int(num) for num in numbers_str.split(',')]
                metric = (set(numbers) == set(ans))
                return metric
        except:
            pass
        return -1