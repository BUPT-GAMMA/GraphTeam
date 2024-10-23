from typing import Dict
from graphteam.roster import Roster

class ChatEnv:
    def __init__(self, with_memory: bool = False):
        self.with_memory = with_memory
        self.roster: Roster = Roster()
        self.env_dict = {}

    def recruit(self, agent_name: str):
        self.roster._recruit(agent_name)

    def exist_employee(self, agent_name: str) -> bool:
        return self.roster._exist_employee(agent_name)

    def print_employees(self):
        self.roster._print_employees()

    def update_codes(self, generated_content):
        self.codes._update_codes(generated_content)

    def rewrite_codes(self, phase_info=None) -> None:
        self.codes._rewrite_codes(self.config.git_management, phase_info)

    def get_codes(self) -> str:
        return self.codes._get_codes()
