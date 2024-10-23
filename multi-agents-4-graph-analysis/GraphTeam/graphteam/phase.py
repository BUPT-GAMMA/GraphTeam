import os
import re
from abc import ABC, abstractmethod

from camel.agents import *
from graphteam.chat_env import ChatEnv
    

class Phase(ABC):

    def __init__(self,
                 assistant_role_name,
                 phase_prompt,
                 role_prompts,
                 phase_name,
                 model_name,
                 rag_config: Optional[RAGConfig] = None,
           
                 ):
        """

        Args:
            assistant_role_name: who receives chat in a phase
            phase_prompt: prompt of this phase
            role_prompts: prompts of all roles
            phase_name: name of this phase
            model_name: name of the model
            rag_config: RAG configuration
        """
        self.assistant_role_name = assistant_role_name
        self.phase_prompt = phase_prompt
        self.phase_env = dict()
        self.phase_name = phase_name
        self.assistant_role_prompt = role_prompts[assistant_role_name]
        self.model_name = model_name
        self.rag_config = rag_config  # 新增 rag_config 支持

    def chatting(
            self,
            messages,
            phase_env = None,
    ) -> str:
        """

        Args:
            messages: list of messages
            memory: memory for the chatting

        Returns:

        """
        conclusion, phase_env = self.agent.generate_response(messages, phase_env)
        return conclusion, phase_env
    @abstractmethod
    def generate_messages(self):
        pass

    @abstractmethod
    def update_phase_env(self, chat_env):
        """
        update self.phase_env (if needed) using chat_env, then the chatting will use self.phase_env to follow the context and fill placeholders in phase prompt
        must be implemented in customized phase
        the usual format is just like:
        ```
            self.phase_env.update({key:chat_env[key]})
        ```
        Args:
            chat_env: global chat chain environment

        Returns: None

        """
        pass

    @abstractmethod
    def update_chat_env(self, chat_env) -> ChatEnv:
        """
        update chan_env based on the results of self.execute, which is self.seminar_conclusion
        must be implemented in customized phase
        the usual format is just like:
        ```
            chat_env.xxx = some_func_for_postprocess(self.seminar_conclusion)
        ```
        Args:
            chat_env:global chat chain environment

        Returns:
            chat_env: updated global chat chain environment

        """
        pass

    def execute(self, chat_env, phase_item) -> ChatEnv:
        """
        execute the chatting in this phase
        1. receive information from environment: update the phase environment from global environment
        2. execute the chatting
        3. change the environment: update the global environment using the conclusion
        Args:
            chat_env: global chat chain environment
            phase_item: configuration of this phase

        Returns:
            chat_env: updated global chat chain environment using the conclusion from this phase execution

        """
        self.update_phase_env(chat_env, phase_item)
        messages = self.generate_messages()
        self.conclusion, self.phase_env = self.chatting(messages, phase_env=self.phase_env)
        chat_env = self.update_chat_env(chat_env)
        return chat_env


class DemandAnalysis(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = DemandAnalysisAgent(model=self.model_name)

    def generate_messages(self):
        phase_prompt = self.phase_prompt.format(task = self.phase_env['question'], 
                                                assistant_role = self.assistant_role_name)
        messages = [
            {"role": "system", "content": self.assistant_role_prompt},
            {"role": "user", "content": phase_prompt}
        ]
        return messages

    def update_phase_env(self, chat_env, phase_item):
        # 从 chat_env 中获取 question 并更新 phase_env
        self.phase_env.update({
            "model": phase_item['model'],
            "question": chat_env.env_dict['question']  # 获取 question
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['Input'] = self.conclusion['Input']
        chat_env.env_dict['Problem'] = self.conclusion['Problem']
        chat_env.env_dict['Output_Format'] = self.conclusion['Output_Format']
        chat_env.env_dict['Graph_Type'] = self.conclusion['Graph_Type']
        return chat_env

class Search(Phase):
    def __init__(self, **kwargs):
        rag_config = kwargs.get('rag_config')

        
        # 检查是否获取到 rag_config 
        if rag_config is None:
            print("Error: rag_config not provided!")

        
        self.agent = SearchAgent(rag_config=rag_config)
        super().__init__(**kwargs)

    def generate_messages(self):
        query = self.phase_env.get('query', '')

        phase_prompt = self.phase_prompt.format(query=query, assistant_role=self.assistant_role_name)
        messages = [
            {"role": "system", "content": self.assistant_role_prompt},
            {"role": "user", "content": phase_prompt}
        ]
        return messages

    def update_phase_env(self, chat_env, phase_item):
        # 更新 phase_env，确保 RAG 数据和 query 传递给 SearchAgent
        self.phase_env.update({
            "model": phase_item['model'],
            "query": str(chat_env.env_dict['Problem']) + " with graph type: " + str(chat_env.env_dict['Graph_Type']),  # 从 DemandAnalysis 的输出中获取查询内容
            "rag_data": chat_env.env_dict['rag_data'],  # 从 chat_env 获取 RAG 数据
            "memory_index": chat_env.env_dict['memory_index'],  # 从 chat_env 获取 memory_index
            "memory_data": chat_env.env_dict['memory_data'],  # 从 chat_env 获取 memory_data
            "library": chat_env.env_dict.get('library', None)  # 获取缓存的 RAG 结果
        })

    def update_chat_env(self, chat_env) -> ChatEnv:
        # Step 1: 检查是否已有缓存的 all_libraries
        all_libraries = self.phase_env.get('all_libraries', None)
        
        chat_env.env_dict['all_libraries'] = all_libraries
        # 如果有缓存的 RAG 结果，直接使用它
        if self.phase_env['library']:
            print("Using cached RAG result")
            print(self.phase_env['library'])
            chat_env.env_dict['Search_Result'] = self.phase_env['library']
        else:
            
            chat_env.env_dict['Search_Result'] = self.conclusion['result']
            # print("+1 Using cached RAG result"+chat_env.env_dict['Search_Result'])
        return chat_env


class Coding(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = CodeAgent(model=self.model_name)

    def generate_messages(self):
        # 从 phase_env 获取 library 并作为提示
        library = self.phase_env.get('library', None)
        if library:
            phase_prompt = self.phase_prompt.format(input=self.phase_env['Input'],
                                                    problem=self.phase_env['Problem'],
                                                    output_format=self.phase_env['Output_Format'],
                                                    search_result=library,  # 使用缓存的 RAG 结果
                                                    Graph_Type=self.phase_env['Graph_Type'],
                                                    assistant_role=self.assistant_role_name)

        else:
            # 没有缓存则正常生成
            phase_prompt = self.phase_prompt.format(input=self.phase_env['Input'],
                                                    problem=self.phase_env['Problem'],
                                                    output_format=self.phase_env['Output_Format'],
                                                    search_result="", 
                                                    Graph_Type=self.phase_env['Graph_Type'], # 空字符串
                                                    assistant_role=self.assistant_role_name)

        messages = [
            {"role": "system", "content": self.assistant_role_prompt},
            {"role": "user", "content": phase_prompt}
        ]
        import json

        # 将 phase_prompt 写入 txt 文件
        with open('output.json', 'w', encoding='utf-8') as json_file:
            json.dump(messages, json_file, ensure_ascii=False, indent=4)

        return messages

    def update_phase_env(self, chat_env, phase_item):
        self.phase_env.update({"model": phase_item['model']})
        self.phase_env.update({"Input": chat_env.env_dict['Input'],
                               "Problem": chat_env.env_dict['Problem'],
                               "Graph_Type":chat_env.env_dict['Graph_Type'],
                               "Output_Format": chat_env.env_dict['Output_Format']})

        # 复用缓存的 RAG 结果
        self.phase_env.update({"library": chat_env.env_dict.get('Search_Result', '')})

    def update_chat_env(self, chat_env) -> ChatEnv:
        library = self.phase_env.get('library', None)
        phase_prompt = self.phase_prompt.format(input=self.phase_env['Input'],
                                                problem=self.phase_env['Problem'],
                                                output_format=self.phase_env['Output_Format'],
                                                search_result=library if library else "",  # 使用缓存的 RAG 结果
                                                Graph_Type=self.phase_env['Graph_Type'],
                                                assistant_role=self.assistant_role_name)
        chat_env.env_dict['Messages'] = [
            {"role": "user", "content": phase_prompt},
            {"role": "assistant", "content": self.conclusion}
        ]
        chat_env.env_dict['Codes'] = self.conclusion
        return chat_env



    
class Execution(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = ExecuteAgent(model=self.model_name)

    def generate_messages(self):
        messages = self.phase_env['Messages']
        messages.append({"role": "system", "content": self.assistant_role_prompt})
        self.phase_env['phase_prompt'] = self.phase_prompt
        return messages

    def update_phase_env(self, chat_env, phase_item):
        self.phase_env.update({"max_retry": phase_item['max_retry'],
                               "model": phase_item['model']})
        self.phase_env.update({"Messages": chat_env.env_dict['Messages'],
                               "Codes": chat_env.env_dict['Codes'],
                               "lock": chat_env.env_dict['lock']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['Messages'] = self.phase_env['Messages']
        if self.conclusion.startswith("Success: "):
            chat_env.env_dict['Output'] = self.conclusion.split("Success: ")[1]
            chat_env.env_dict['run'] = True
        else:
            chat_env.env_dict['Output'] = self.conclusion
            chat_env.env_dict['run'] = False
        return chat_env
    
class Reasoning(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = ReasoningAgent(model=self.model_name)

    def generate_messages(self):
        phase_prompt = self.phase_prompt.format(assistant_role=self.assistant_role_name,
                                                input=self.phase_env['Input'],
                                                problem=self.phase_env['Problem'],
                                                output_format=self.phase_env['Output_Format'])
        messages = [
            {"role": "system", "content": self.assistant_role_prompt},
            {"role": "user", "content": phase_prompt}
        ]
        return messages

    def update_phase_env(self, chat_env, phase_item):
        self.phase_env.update({"model": phase_item['model']})
        self.phase_env.update({"run": chat_env.env_dict['run'],
                                "Input": chat_env.env_dict['Input'],
                                "Problem": chat_env.env_dict['Problem'],
                                "Output_Format": chat_env.env_dict['Output_Format']})
                              

    def update_chat_env(self, chat_env) -> ChatEnv:
        if self.conclusion!= None:
            chat_env.env_dict['Output'] = self.conclusion
        return chat_env

#纠正输出格式
class Correction(Phase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent = CorrectionAgent(model=self.model_name)

    def generate_messages(self):
        messages = [
            {"role": "system", "content": self.assistant_role_prompt},
        ]
        return messages

    def update_phase_env(self, chat_env, phase_item):
        self.phase_env.update({"model": phase_item['model'],
                               "max_retry": phase_item['max_retry'],
                               "phase_prompt": self.phase_prompt,
                               "assistant_role": self.assistant_role_name})
        self.phase_env.update({"Output": chat_env.env_dict['Output'],
                               "Output_Format": chat_env.env_dict['Output_Format']})

    def update_chat_env(self, chat_env) -> ChatEnv:
        chat_env.env_dict['Output'] = self.conclusion
        return chat_env