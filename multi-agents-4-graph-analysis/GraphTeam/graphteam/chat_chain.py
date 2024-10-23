import importlib
import json
import logging
import os
import shutil
import time
from datetime import datetime
from camel.configs import ChatGPTConfig
from camel.rag_configs import RAGConfig
from graphteam.chat_env import ChatEnv

def check_bool(s):
    return s.lower() == "true"


class ChatChain:

    def __init__(self,
                 config_path: str = None,
                 config_phase_path: str = None,
                 config_role_path: str = None,
                 question: str = None,
                 model_name: str = "gpt-4o-mini",
                 rag_data: list = None,
                 rag_config: RAGConfig = None,
                 memory_index=None,
                 memory_data=None,
                 library:str= None) -> None:
        """

        Args:
            config_path: path to the ChatChainConfig.json
            config_phase_path: path to the PhaseConfig.json
            config_role_path: path to the RoleConfig.json
            question: user input question
            model_name: model name
            rag_data: RAG data
            rag_config: RAGConfig
            memory_index: memory index
            memory_data: memory data
            library: library data
        """

        # load config file
        self.config_path = config_path
        self.config_phase_path = config_phase_path
        self.config_role_path = config_role_path


        with open(self.config_path, 'r', encoding="utf8") as file:
            self.config = json.load(file)
        with open(self.config_phase_path, 'r', encoding="utf8") as file:
            self.config_phase = json.load(file)
        with open(self.config_role_path, 'r', encoding="utf8") as file:
            self.config_role = json.load(file)

        # init chatchain config and recruitments
        self.chain = self.config["chain"]
        self.recruitments = self.config["recruitments"]
        # self.web_spider = self.config["web_spider"]

        # init ChatEnv                                             
        self.chat_env = ChatEnv()

        # the user input prompt will be self-improved (if set "self_improve": "True" in ChatChainConfig.json)
        # the self-improvement is done in self.preprocess
        self.question_raw = question
        self.chat_env.env_dict["question"] = self.question_raw

        # RAG 数据，存储在 chat_env 中
        self.chat_env.env_dict["rag_data"] = rag_data  

        self.chat_env.env_dict['memory_index']=memory_index

        self.chat_env.env_dict['memory_data']=memory_data
               
        self.chat_env.env_dict["library"] = library 


        self.rag_config = RAGConfig()

        # init role prompts
        self.role_prompts = dict()
        for role in self.config_role:
            self.role_prompts[role] = "\n".join(self.config_role[role])

        self.model_name = model_name

        # init Phase instances
        self.phase_module = importlib.import_module("graphteam.phase")
        self.phases = dict()
        for phase in self.config_phase:
            assistant_role_name = self.config_phase[phase]['assistant_role_name']
            phase_prompt = "\n\n".join(self.config_phase[phase]['phase_prompt'])
            phase_class = getattr(self.phase_module, phase)
            phase_instance = phase_class(assistant_role_name=assistant_role_name,
                                         phase_prompt=phase_prompt,
                                         role_prompts=self.role_prompts,
                                         phase_name=phase,
                                         model_name=self.model_name,
                                         rag_config=self.rag_config, # 将 RAG 配置传递给 phase
                                        )
            self.phases[phase] = phase_instance

    def make_recruitment(self):
        """
        recruit all employees
        Returns: None

        """
        for employee in self.recruitments:
            self.chat_env.recruit(agent_name=employee)

    def execute_step(self, phase_item: dict):
        """
        execute single phase in the chain
        Args:
            phase_item: single phase configuration in the ChatChainConfig.json

        Returns:

        """

        phase = phase_item['phase']
        if phase in self.phases:
            self.chat_env = self.phases[phase].execute(self.chat_env, phase_item)
        else:
            raise RuntimeError(f"Phase '{phase}' is not yet implemented in chatdev.phase")

    def execute_chain(self,lock):
        """
        execute the whole chain based on ChatChainConfig.json
        lock: lock object
        Returns: None

        """
        self.chat_env.env_dict["lock"] = lock
        for phase_item in self.chain:
            self.execute_step(phase_item)
