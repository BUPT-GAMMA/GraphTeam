from .base import DyGraphTask
import numpy as np
import re

    
    
class DyGraphTaskWhatNode(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)
        
        # select time
        times = list(set(context[:,2].tolist()))
        t = int(np.random.choice(times))
        
        # select n1
        con_t = context[context[:, 2] == t]
        nodes = list(set(list(con_t[:, :2].flatten())))
        n1 = int(np.random.choice(nodes))
        
        # give answers
        answer = set()
        for e1,e2,_ in con_t:
            if n1 == e1 :
                answer.add(int(e2))
            if n1 == e2:
                answer.add(int(e1))
        answer = list(answer)
        context = context.tolist()
        qa = {
            "context": context,
            "query": [n1, t],
            "answer": answer,
            "task": self.task
        }
        try:
            assert len(answer) > 0
        except Exception as e:
            import pdb; pdb.set_trace()
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer what nodes are linked with a given node at a given time in the dynamic graph.\n"
       
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(1, 2, 1), (1, 3, 1), (1, 2, 5), (3, 1, 6)],
                [1, 1], 
                [2, 3]
             ]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"What nodes are linked with node {query[0]} at time {query[1]}?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
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
        
        match = re.search(r"""at time [\d]+ are\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        
        match = re.search(r"""at time [\d]+ is\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        
        match = re.search(r"""[O|o]utput\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        
        match = re.search(r"""[R|r]eturn\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        
        match = re.search(r"""[L|l]ist\s*[:`'"]?\s*\[([\d,\s]+)\]""", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        return -1