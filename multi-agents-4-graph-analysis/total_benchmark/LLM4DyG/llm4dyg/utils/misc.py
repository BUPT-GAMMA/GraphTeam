import os
import shutil
def remove_dir(log_dir):
    log_dir = os.path.abspath(log_dir)
    print(f"Removing {log_dir} ? y or n ")
    a = input()
    if a.strip().lower() == "y":
        print(f"Removing {log_dir}")
        shutil.rmtree(log_dir)

def format(x):
    if not isinstance(x,str):
        return f"{x*100:.1f}"
    return x

import time
class TPMController:
    def __init__(self, tpm = 90000, min_token = 5000, start_token = 90000) -> None:
        self.tpm = tpm
        self.start_time = time.time()
        self.token_left = start_token
        self.min_token = min_token
        self.token_per_second = self.tpm / 60
    
    def time_token(self):
        self.token_left += (time.time()-self.start_time)* self.token_per_second
        self.token_left = min(self.token_left, self.tpm)
        
    def use_token(self, token):
        self.token_left -= token

    def get_token(self):
        if self.token_left <= self.min_token:
            sleep_time = self.min_token/self.token_per_second
            print(f"token_left: {self.token_left}, sleep: {sleep_time}")
            time.sleep(sleep_time)
            self.token_left += self.min_token
        self.start_time = time.time()
        return self.token_left