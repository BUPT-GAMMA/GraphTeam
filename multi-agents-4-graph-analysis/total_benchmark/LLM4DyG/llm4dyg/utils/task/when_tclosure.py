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

def find_triad_time(context, triad):
    ts = []
    ns = [[0,1],[1,2],[0,2]]
    for i in range(len(ns)):
        n1, n2 = ns[i]
        n1, n2 = triad[n1], triad[n2]
        t = find_edge_t(context, (n1,n2), 0)
        assert t >= 0, "no answer"
        ts.append(t) 
    t = max(ts)
    return t

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
    return triangles

class DyGraphTaskWhenTClosure(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        # select triad
        edges = [(x[0], x[1]) for x in context]
        triads = find_triangles(edges)
        np.random.shuffle(triads)
        triad = triads[0]
        
        # get answer
        time = int(find_triad_time(context, triad))
        
        query = list(map(int, triad))
        answer = time

        context = context.tolist()
        qa = {
            "context": context,
            "query": query,
            "answer": answer,
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer when the three nodes in the dynamic graph first close the triad. Two nodes with a common neighbor is said to have a triadic closure, if they are linked since some time so that the three nodes have linked with each other to form a triad.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as an integer number at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 2, 0), (2, 3, 1), (1, 2, 2), (3, 1, 3)],
                [0, 2 ,3],
                3
            ],
            [
                [(0, 2, 0), (2, 3, 1), (1, 2, 2), (3, 1, 5)],
                [0, 2 ,3],
                5
            ],
            [
                [(0, 2, 0), (2, 3, 1), (1, 2, 2), (3, 1, 0)],
                [0, 2 ,3],
                1
            ],
            [
                [(0, 2, 0), (2, 3, 5), (1, 2, 2), (3, 1, 0)],
                [0, 2 ,3],
                5
            ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"When did the three nodes {query} first close the triad?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"Answer:\s*(\d+)\s*", response)
        # import pdb; pdb.set_trace()
        if match:
            answer = int(match.group(1))
            return answer == ans
        else:
            match = re.search(r"answer is:\s*(\d+)\s*", response)
            if match:
                answer = int(match.group(1))
                return answer == ans
            else:
                match = re.search(r"at time\s*(\d+)\s*", response)
                if match:
                    answer = int(match.group(1))
                    return answer == ans
                else:
                    return -1