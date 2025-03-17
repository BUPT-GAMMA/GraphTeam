import logging
from uuid import uuid4
from tqdm import tqdm
import pandas as pd
import os
import time

class TaskLogger:
    """任务专属日志记录器"""
    def __init__(self, task_id, log_dir="logs"):
        self.task_id = task_id
        self.log_dir = log_dir
        self.logger = logging.getLogger(f"task_{task_id}")
        self.logger.setLevel(logging.INFO)
        
        # 创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        
        # 文件处理器
        self.file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{task_id}.log"),
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, level, message):
        self.logger.log(level, f"[{self.task_id}] {message}")

    def close(self):
        """关闭文件处理器"""
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()