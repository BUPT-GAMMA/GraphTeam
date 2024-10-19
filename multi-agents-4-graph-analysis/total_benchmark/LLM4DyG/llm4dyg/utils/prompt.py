
def get_imp(imp):
    if imp == 0:
        return f""
    elif imp == 24:
        return f'Pick time and then nodes. \n'
    elif imp == 25:
        return f'Pick nodes and then time. \n'
    elif imp == 26:
        return f"Take a deep breath and work on this problem step-by-step. \n"
    elif imp == 27:
        return f'Think nodes and then time. \n'
    elif imp == 28:
        return f'Think time and then nodes. \n'
    else:
        raise NotImplementedError()
    
import types

class DyGraphPrompt:
    def __init__(self, obj_task, args) -> None:
        add_cot = args.add_cot
        add_role = args.add_role
        num_examplars = args.num_examplars
        dyg_type = args.dyg_type
        
        self.instructor_role = "You are an excellent dynamic network analyzer, with a good understanding of the structure of the graph and its evolution through time. \n"
        if dyg_type == 0:
            self.instructor_dyg = f"A dynamic graph is represented as a list of tuples, where each tuple (u, v, t) denotes that there is an edge at time t between node u and node v. For example, (6, 5, 2) denotes that node 6 is linked with node 5 at time 2. \n"
        elif dyg_type == 1:
            self.instructor_dyg = f"In an undirected dynamic graph, (u, v, t) means that node u and node v are linked with an undirected edge at time t.\n"
        else:
            raise NotImplementedError(f"dyg_type {dyg_type} not implemented")
        
        self.args = args
        if args:
            imp = self.args.__dict__.get("imp", 0)
        else:
            imp = 0
        self.prompt_imp = get_imp(imp)
        self.prompt_cot = f"You can think it step by step.\n"
        self.add_cot = add_cot
        self.add_role = add_role
        self.num_examplars = num_examplars
        self.obj_task = obj_task
        
    def generate_prompt_qa(self, context, query = None, answer = None, *args, **kwargs):
        # generate prompt components
        instructor_role, instructor_dyg = self.instructor_role if self.add_role else "", self.instructor_dyg
        prompt_cot = self.prompt_cot if self.add_cot else ""
        
        prompt_context = self.obj_task.generate_context_prompt(context)
        instructor_task = self.obj_task.generate_instructor_task()
        instructor_answer = self.obj_task.generate_instructor_answer()
        prompt_examplars = self.obj_task.generate_prompt_examplars(self.num_examplars) if self.num_examplars else ""
        prompt_question = self.obj_task.generate_prompt_question(query)

        prompt_seq = [
            instructor_role,
            instructor_dyg,
            instructor_task,
            self.prompt_imp,
            instructor_answer,
            #prompt_examplars,
            prompt_context,
            prompt_question,
            prompt_cot
        ]
        
        if self.args:
            short = self.args.__dict__.get("short", 0)
            if short==1:
                prompt_seq.append('Give a short answer.')
            elif short==2:
                prompt_seq.append('Note that the time represents year, month, day, for example, 20200925 means 25th day in September in 2020, and 19990102 < 20200925 < 20231207')
            elif short==3:
                prompt_seq.append('Note that the time represents unix timestamp, for example, 1348839350 < 1476979078 < 1547036558')
        
        prompt = "".join(prompt_seq)
        
        prompt_qa = {
            "prompt": prompt,
            "answer": answer,
        }
        return prompt_qa