from .base import DyGraphTask
import numpy as np
import re

class DyGraphTaskNumTime(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        answer = len(list(set(np.array(context)[:, 2])))
        qa = {
            "context": context,
            "query": [],
            "answer": answer,
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to answer the number of timestamps in the dynamic graph.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as an integer number at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [[(7, 4, 1), (7, 0, 2), (7, 1, 0), (7, 5, 2)], [], 3],
            [[(5, 3, 0), (3, 5, 0), (5, 2, 1), (5, 3, 1)], [], 2]
        ]
        return self.make_qa_example(num, qa)
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"What is the number of timestamps in the dynamic graph?\n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        match = re.search(r"Answer:\s*(\d+)\s*", response)
        if match:
            answer = int(match.group(1))
            return answer == ans
        else:
            return -1