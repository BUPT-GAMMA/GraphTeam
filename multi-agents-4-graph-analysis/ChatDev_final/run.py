import threading
from tqdm import tqdm
import pandas as pd
import os
import sys
import json
import time
import openai
import argparse
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt

# answer_format_dict = {
#     "matching": "In the answer, you should replace number with the actual problem and result,Output format such as : applicant 0: job 2 \n 1 applicants can find the job they are interested in.",
#     "shortest_path": "In the answer, you should replace number with the actual problem and result, Output format such as : The shortest path from node 1 to node 6 is 1,4,6 with a total weight of 5 ",
#     "topology": "In the answer, you should replace number with the actual problem and result, below is just an example, Output format such as : The solution is: 2,3,7",
#     "cycle": " Output format such as : TRUE or False",
#     "GNN": "In the answer, you should replace number with the actual problem and result, Output format such as : \n node x: [1,1]\n ",
#     "hamilton": "In the answer, you should replace number with the actual problem and result, below is just an example, Output format such as : Yes. The path can be: 1,4,8",
#     "flow": "In the answer, you should replace number with the actual problem and result, Output format such as : The maximum flow from node 2 to node 6 is 3",
#     "connectivity": "Output format such as : TRUE or False"
# }


lock = threading.Lock()
os.environ['OPENAI_API_KEY'] = 'sk-566CUIFlZOj6aopT3e66D624E3004dB3881fF8F4Fa86AcFf'
os.environ['OPENAI_BASE_URL'] = "https://dzqc.link/v1/"
os.environ['OPENAI_API_BASE'] = 'https://dzqc.link/v1/'
root = os.path.dirname(__file__)
sys.path.append(root)






from chatdev.chat_chain import ChatChain

try:
    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")
    
def load_rag_data(root, json_filenames=None):
    """
    加载多个 RAG JSON 文件，并将它们的数据合并。
    
    Args:
        root: 项目的根目录
        json_filenames: JSON 文件名列表。
    
    Returns:
        一个包含所有 JSON 文件内容的知识库列表。
    """
    knowledge_base = []
    if json_filenames is None:
        json_filenames = ["networkx_reference.json"]

    # 拼接根目录和文件路径
    for json_file in json_filenames:
        file_path = os.path.join(root, "data", json_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_base.extend(json.load(f))  # 将文件中的内容扩展到 knowledge_base
        else:
            print(f"Warning: JSON file {file_path} not found.")
    
    return knowledge_base

def load_memory_data(json_file):
    """
    从 JSON 文件中加载记忆数据。
    
    Args:
        json_file (str): JSON 文件路径。
    
    Returns:
        记忆数据。
    """
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            memory_data = json.load(f)
    else:
        memory_data = []
    
    return memory_data

def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "Config", company)
    default_config_dir = os.path.join(root, "Config", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)

import json

def csv2json(total_data, output_path):
    # Convert the DataFrame to a dictionary (with lists of column values)
    data_dict = total_data.to_dict(orient='records')  # 'records' format makes each row a dictionary

    # Write the JSON data to the output file with indentation
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

        



# 假设你有多个 JSON 文件
json_files = ["filtered_networkx_reference_edition.json"]

# 加载所有 RAG JSON 数据
rag_data = load_rag_data(root, json_filenames=json_files)
config_path, config_phase_path, config_role_path = get_config("Default")
model = "gpt-4o-mini"
memory_index=load_memory_data("ChatDev_final/memory/memory_index.json")
# print(memory_index)
memory_data=load_memory_data("ChatDev_final/memory/memory_info.json")


def run_threaded(start_idx, end_idx, progress_bar, category_data, index):
    """
    用于处理问题的函数，将智能体系统的输出存储到 result 列中
    """
    for i in range(start_idx, end_idx):

        question = category_data['question'][i]
        library = category_data['search_result'][i]
             # 判断 library 是否为 NaN
        if pd.isna(library):
            library = None
        # print(library)
        # 多智能体系统生成 reply
        chat_chain = ChatChain(config_path=config_path,
                            config_phase_path=config_phase_path,
                            config_role_path=config_role_path,
                            question=question,
                            model_name=model,
                            rag_data=rag_data,
                            memory_index=memory_index,
                            memory_data=memory_data,
                            library=library)
        chat_chain.make_recruitment()
        chat_chain.execute_chain(lock)

        # 将多智能体系统的输出结果赋值给 result 列
        i = index[i]
        total_data.at[i, 'result'] = chat_chain.chat_env.env_dict["Output"]  # 用 .at 确保正确赋值
        total_data.at[i, 'run'] = chat_chain.chat_env.env_dict["run"]
        total_data.at[i, 'code'] = chat_chain.chat_env.env_dict["Codes"]
        if library is None:
            total_data.at[i, 'search_result'] = chat_chain.chat_env.env_dict["Search_Result"]
        # print(total_data.at[i, 'search_result'])
        # 每次记录生成的代码和其他信息
        print(f"Generated reply for row {i}: {total_data.at[i, 'result']}")

        progress_bar.update(1)

def run_all(category_data, index,library=None):
    """
    多线程运行数据处理，生成 reply 列，并保存到 modified 文件
    """
    num_threads = 1
    threads = []

    chunk_size = (len(category_data) + num_threads - 1) // num_threads

    global_progress_bar = tqdm(total=len(category_data), desc="Total Progress")

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(category_data))

        thread = threading.Thread(target=run_threaded, args=(start_idx, end_idx, global_progress_bar, category_data, index))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    global_progress_bar.close()

def run_one(question, index):
    """
    用于单独处理一个问题的函数
    """
    chat_chain = ChatChain(config_path=config_path,
                           config_phase_path=config_phase_path,
                           config_role_path=config_role_path,
                           question=question,
                           model_name=model,
                           rag_data=rag_data,
                           memory_index=memory_index,
                           memory_data=memory_data,
                           library=None)
    chat_chain.make_recruitment()
    chat_chain.execute_chain(lock)
    total_data.at[index, 'result'] = chat_chain.chat_env.env_dict["Output"]
    library=chat_chain.chat_env.env_dict.get("Search_Result")

    return library

# 读取data中type列的值，找到所有不同的值，和值对应的index
def get_categories(data):
    """
    获取数据中 type 列的所有不同值
    """
    #如果不存在type列，为所有数据添加type列
    if 'type' not in data.columns:
        data['type'] = "default"
    return data['type'].unique()

def get_category_data(data, category):
    """
    获取数据中某个 category 的所有数据
    """
    return data[data['type'] == category], data[data['type'] == category].index

def get_categories_finished(data):
    
    filtered_df = data[data['result'].notna()]
    
    return filtered_df['type'].unique()

file_path = 'total_benchmark/GNN_benchmark/GNN_benchmark_test.json'
output_path = 'total_benchmark/GNN_benchmark/result.json'
#total_data = pd.read_csv(file_path, dtype={'search_result': 'object'})
total_data = pd.read_json(file_path, orient='records', dtype={'search_result': 'object'})[:2]
#total_data = pd.DataFrame(total_data)
total_data['result'] = None
total_data['run'] = None
total_data['code'] = None
total_data['search_result'] = None

categories = get_categories(total_data)
for category in categories:
    finished_file_path = Path(output_path)  
    # import pdb
    # pdb.set_trace()

    if not finished_file_path.exists():
        categories_finished = []
    else:
        finished_data = pd.read_json(output_path)
        total_data['result'] = finished_data['result']
        total_data['run'] = finished_data['run']
        total_data['code'] = finished_data['code']
        # 检查并添加缺失的列
        if 'search_result' not in finished_data.columns:
            finished_data['search_result'] = None
        total_data['search_result']= finished_data['search_result'].astype(object)
        categories_finished = get_categories_finished(finished_data)

    if category in categories_finished:
        continue

    print(f"Processing category {category}")

    category_data, category_index = get_category_data(total_data,category)

    category_data = category_data.reset_index(drop=True)

    run_all(category_data, category_index)
    # 保存结果
    csv2json(total_data, output_path)






