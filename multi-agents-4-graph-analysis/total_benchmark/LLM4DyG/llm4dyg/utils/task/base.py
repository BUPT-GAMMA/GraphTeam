import re
class DyGraphTask:
    def __init__(self, task, args) -> None:
        self.task = task
        self.args = args
         
    def generate_qa(self, info, *args, **kwargs):
        pass
    
    def generate_instructor_task(self, *args, **kwargs):
        pass
    
    def generate_instructor_answer(self, *args, **kwargs):
        return "Give the answer as a python list at the last of your response after 'Answer:'.\n"
    
    def generate_prompt_examplars(self, num, *args, **kwargs):
        pass
    
    def generate_prompt_question(self, query = None, *args, **kwargs):
        pass
    
    def generate_context_prompt(self, context):
        edge_type = self.args.__dict__.get("edge_type", 0)
        if edge_type == 0:
            return f"Question: Given an undirected dynamic graph with the edges {[tuple(x) for x in context]}. "
        elif edge_type == 1:
            edge_desc = ""
            for x in context:
                edge_desc += f"{tuple(x)} "
            return f"Question: Given an undirected dynamic graph with the edges {edge_desc}. "
        else:
            raise NotImplementedError(f"edge_type {edge_type} not implemented")

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
    def make_qa_example(self, num, qa):
        if num == 0:
            return ""
        examples = []
        for c,q,a in qa:
            example = f"{self.generate_context_prompt(c)}{self.generate_prompt_question(q)}Answer:{a}\n"
            examples.append(example)
        
        if num == 1:
            prompt = "\n Here is an example: \n" + "\n".join(examples[:num])
        else:
            prompt = f"\n Here are {num} examples: \n" + "\n".join(examples[:num])
        return prompt
        

   