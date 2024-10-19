from .base import DyGraphTask
import numpy as np
import re
import networkx as nx
def find_neighbors(x, con):
    res = set()
    for e1,e2,t in con:
        if e1 == x:
            res.add(e2)
        if e2 == x:
            res.add(e1)
    return list(res)

def judge_connect(es, n1, n2):
    ts = sorted(list(set(list(es[:, 2]))))
    G = nx.Graph() 
    for t in ts:
        e = es[es[:, 2] == t][:, :2]
        G.add_edges_from(e)

        if nx.has_path(G, n1, n2):
            return int(t)
    assert False, "No path found"

    
class DyGraphTaskWhenConnect(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        answer = set()
        
        # select n1
        nodes = list(set(list(context[:, :2].flatten())))
        n1 = int(np.random.choice(nodes))
        
        # select n2
        n1nodes = find_neighbors(n1, context)
        n2 = int(np.random.choice(n1nodes))

        # give answer
        answer = judge_connect(context, n1, n2)

        context = context.tolist()
        qa = {
            "context": context,
            "query": [n1, n2],
            "answer": answer,
            "task": self.task
        }

        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer when two nodes are first connected in the dynamic graph. Two nodes are connected if there exists a path between them.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as an integer number at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 1, 0), (1, 2, 1), (0, 2, 2)],
                [0, 2], 
                1
             ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"When are node {query[0]} and node {query[1]} first connected?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"""Answer\s*[:`'"]?\s*(\d+)\s*""", response)
        if match:
            answer = int(match.group(1))
            return answer == ans
        
        match = re.search(r"""answer is\s*[:`'"]?\s*(\d+)\s*""", response)
        if match:
            answer = int(match.group(1))
            return answer == ans
        
        match = re.search(r"""at time\s*[:`'"]?\s*(\d+)\s*""", response)
        if match:
            answer = int(match.group(1))
            return answer == ans
        return -1