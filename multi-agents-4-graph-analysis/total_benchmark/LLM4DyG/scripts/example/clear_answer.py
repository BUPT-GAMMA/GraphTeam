#清除某个文件夹及其子文件夹下的所有名为answer_gpt-4o-mini.json的文件
import os
import shutil

def remove_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "answer_gpt-4o-mini.json":
                os.remove(os.path.join(root, file))
        for dir in dirs:
            remove_files(os.path.join(root, dir))
          
remove_files("total_benchmark/LLM4DyG/logs_n=5")
remove_files("total_benchmark/LLM4DyG/logs_n=10")
remove_files("total_benchmark/LLM4DyG/logs_n=20")
