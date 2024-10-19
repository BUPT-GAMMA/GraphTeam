from .base import DyGraphTask
import numpy as np
import re

class DyGraphTaskNumEdge(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)
        time = np.max(context[:, 2]) + 1
        answer = [int(sum(context[:, 2] == t)) for t in range(time)]
        context = context.tolist()
        qa = {
            "context": context,
            "query": [int(time - 1)],
            "answer": answer,
            "task": self.task
        }
        try:
            assert len(answer) > 0
        except Exception as e:
            import pdb; pdb.set_trace()
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer the number of edges at each time in the dynamic graph.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(1, 2, 0), (1, 3, 1), (1, 2, 1), (3, 1, 3), (5, 1, 3)],
                [3] , 
                [1, 2, 0, 2]
             ]
        ]
        return self.make_qa_example(num, qa)
        
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"What are the number of edges in the graph from time 0 to time {query[0]}?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"Answer:\s*\[([\d,\s]+)\]", response)
        if match:
            numbers_str = match.group(1)
            numbers = [int(num) for num in numbers_str.split(',')]
            metric = (set(numbers) == set(ans))
            return metric
        else:
            return -1