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

import networkx as nx
def find_triangles(edges):
    # Create an empty graph
    graph = nx.Graph()

    # Add edges to the graph
    graph.add_edges_from(edges)

    # Find triangles in the graph
    triangles = list(nx.enumerate_all_cliques(graph))

    # Filter out triangles
    triangles = [triangle for triangle in triangles if len(triangle) == 3]
    assert len(triangles) > 0, "no answer"
    # assert len(triangles) > 0, "no answer" + f"for {edges}"
    return triangles

def judge_path(path, triads):
    iters = set([x for x in permutations(path, 3)])
    # print("judge", iters)
    for it in iters:
        if it in triads:
            return True
    return False

def generate_path(path_length, label, num_nodes, triads):
    iters = [x for x in permutations(np.arange(num_nodes), path_length)]
    np.random.shuffle(iters)
    for path in iters:
        # print("gpath",path)
        judge = judge_path(path, triads)
        if label == judge:
            return path
    assert False, "no answer" 
    # assert False, "no answer" + f" for answer {label} \n iters {iters} \n triads {triads}" 

class DyGraphTaskCheckTClosure(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        # select triad
        edges = [(x[0], x[1]) for x in context]
        triads = find_triangles(edges)
        triads = set([tuple(sorted(x)) for x in triads])
        # print("triads",triads)
        
        # select num_nodes
        nodes = list(set(list(context[:, :2].flatten())))
        num_nodes = len(nodes)
        path_length = 3
        assert num_nodes >= path_length, "no answer"
        label = kwargs['label']
        answer = 'yes' if label else 'no'
        # print("ans",answer)
        
        # answer 
        path = generate_path(path_length, label, num_nodes, triads)
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
        return f"Your task is to answer whether three nodes in the dynamic graph formed a closed triad. A closed triad is composed of three nodes which have linked with each other some time. \n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as yes or no at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 2, 0), (1, 2, 1), (1, 0, 2), (3, 1, 3)],
                [0, 2 ,1],
                'yes'
            ],
            [
                [(0, 2, 0), (1, 2, 1), (1, 5, 2), (3, 1, 3)],
                [0, 2 ,1],
                'no'
            ],
            [
                [(0, 2, 0), (1, 2, 1), (1, 5, 2), (5, 2, 3)],
                [5, 2 ,1],
                'yes'
            ],
            [
                [(0, 2, 0), (1, 2, 1), (1, 5, 2), (5, 2, 3)],
                [0, 2 ,1],
                'no'
            ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"Did the three nodes {query} form a closed triad?\n"
    
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