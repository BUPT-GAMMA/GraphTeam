from .base import DyGraphTask
import numpy as np
import re

def judge_ordered(con):
    con = np.array(con)
    for i in range(len(con)-1):
        if con[i,2] > con[i+1,2]:
            return False
    return True
    
class DyGraphTaskSortEdge(DyGraphTask):
    def generate_qa(self, info, *args, **kwargs):
        context = info['edge_index']
        context = np.array(context)
        
        assert len(context)>3, "no answer"
        # shuffle edge
        np.random.shuffle(context)
        
        # sort edge by time 
        context = context.tolist()
        qa = {
            "context": context,
            "query": [],
            "answer": [],
            "task": self.task
        }
        return qa
    
    def generate_instructor_task(self, *args, **kwargs):
        return f"Your task is to sort the edges in the dynamic graph by time from earlest to latest.\n"
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:'.\n"

    def generate_prompt_examplars(self, num, *args, **kwargs):
        qa = [
            [
                [(0, 1, 3), (1, 3, 1), (1, 2, 0)],
                [] , 
                [(1, 2, 0), (1, 3, 1), (0, 1, 3)]
             ]
        ]
        return self.make_qa_example(num, qa)
        
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        return f"Sort the edges in the dynamic graph by time from earliest to latest. \n"
    
    def evaluate(self, qa, response):
        ans = qa['answer']
        context = qa['context']
        def parse_match(match):
            numbers_str = match.group(1)
            numbers = eval(numbers_str)
            ans = np.array(numbers)
            set_con = set([tuple(x) for x in context])
            set_ans = set([tuple(x) for x in ans])
            if set_con != set_ans:
                return False       
            metric = judge_ordered(ans)
            return metric
        try:
            match = re.search(r"Answer:\s*(\[.*?\])", response)
            if match:
                return parse_match(match)
            
            match = re.search(r"""are\s*[:`'"]*\s*(\[.*?\])""", response)
            if match:
                return parse_match(match)
            
            match = re.search(r"""answer is\s*[:`'"]*\s*(\[.*?\])""", response)
            if match:
                return parse_match(match)
                    
            match = re.search(r"Answer.*(\[.*?\])", response)
            if match:
                return parse_match(match)
                
            match = re.search(r"[O|o]utput[^\[\]]*(\[.*\])", response)
            if match:
                return parse_match(match)
        except Exception as e:
            print(e)
            
        return -1