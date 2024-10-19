from .base import DyGraphTask
import numpy as np
import re


class DyGraphTaskLinkPred(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)

        t = np.max(context[:,-1])
        hist = context[context[:,-1]<t]
        future = context[context[:,-1]==t]
        
        answer = set()
        
        # positive edges
        ego_nodes = info['ego_nodes']
        ego_set = set(ego_nodes)
        pos_edges = [(int(x[0]),int(x[1])) for x in future if x[0] in ego_set and x[1] in ego_set]
        neg_edges = []
        pos_set = set(pos_edges)
        assert len(pos_edges) > 0, 'no positive edges'
        
        
        num_node = len(ego_nodes)
        for i in range(num_node):
            for j in range(i+1,num_node):
                n1 = ego_nodes[i]
                n2 = ego_nodes[j]
                if (n1,n2) not in pos_set and (n2,n1) not in pos_set:
                    neg_edges.append((n1,n2))
        
        np.random.shuffle(pos_edges)
        np.random.shuffle(neg_edges)
        pos_edges = pos_edges[:10]
        neg_edges = neg_edges[:10]
        
        query = pos_edges + neg_edges
        answer = [1] * len(pos_edges) + [0] * len(neg_edges)
        idxs = np.arange(len(answer))
        np.random.shuffle(idxs)
        query = [query[i] for i in idxs]
        answer = [answer[i] for i in idxs]
                
        qa = {
            "context": hist.tolist(),
            "query": query,
            "answer": answer,
            "task": self.task
        }
        try:
            assert len(answer) > 0
        except Exception as e:
            import pdb; pdb.set_trace()
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to predict whether the given edges will be linked in the future.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:', use 0 as no, 1 as yes. Tip: the answers for most edges are no.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 1, 0), (0, 2, 1), (0, 3, 2), (4, 2, 2)],
                [(1, 2), (2, 3), (4, 5), (3, 4)], 
                [1, 1, 0, 0]
            ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"Will the following edges be linked in the future? {query}\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"Answer:\s*\[([\d+(\.\d+)?,\s]+)\]", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""answer is\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""at time [\d]+ are\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""at time [\d]+ is\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""[O|o]utput\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""[R|r]eturn\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        
        match = re.search(r"""[L|l]ist\s*[:`'"]?\s*\[([\d+(\.\d+)?,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [float(num) for num in numbers_str.split(',')]
            metric = cal_metric(ans, numbers)
            return metric
        return -1
                                

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def cal_metric(y, pred):
    if len(y) != len(pred):
        return -1
    
    metric = f1_score(y, pred)

    return metric