import threading
from tqdm import tqdm
import pandas as pd
import os
import sys
import json
import argparse
from pathlib import Path

# Import ChatChain after setting the root path
root = os.path.dirname(__file__)
sys.path.append(root)
from graphteam.chat_chain import ChatChain

# Initialize threading lock
lock = threading.Lock()

# Set environment variables
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''
os.environ['OPENAI_BASE_URL'] = ''

answer_format_dict = {
    "matching": "In the answer, you should replace number with the actual problem and result,Output format such as : applicant 0: job 2 \n 1 applicants can find the job they are interested in.",
    "shortest_path": "In the answer, you should replace number with the actual problem and result, Output format such as : The shortest path from node 1 to node 6 is 1,4,6 with a total weight of 5 ",
    "topology": "In the answer, you should replace number with the actual problem and result, below is just an example, Output format such as : The solution is: 2,3,7",
    "cycle": " Output format such as : TRUE or False",
    "GNN": "In the answer, you should replace number with the actual problem and result, Output format such as : \n node x: [1,1]\n ",
    "hamilton": "In the answer, you should replace number with the actual problem and result, below is just an example, Output format such as : Yes. The path can be: 1,4,8",
    "flow": "In the answer, you should replace number with the actual problem and result, Output format such as : The maximum flow from node 2 to node 6 is 3",
    "connectivity": "Output format such as : TRUE or False"
}

# Check OpenAI API version
try:
    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported."
    )

def load_rag_data(root, json_filenames=None):
    """
    Load multiple RAG JSON files and merge their data.

    Args:
        root: Root directory of the project.
        json_filenames: List of JSON filenames.

    Returns:
        A list containing all JSON file contents as the knowledge base.
    """
    knowledge_base = []
    if json_filenames is None:
        json_filenames = ["networkx_reference.json"]

    for json_file in json_filenames:
        file_path = os.path.join(root, "data", json_file)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_base.extend(json.load(f))
        else:
            print(f"Warning: JSON file {file_path} not found.")

    return knowledge_base

def load_memory_data(json_file):
    """
    Load memory data from a JSON file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        Memory data.
    """
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            memory_data = json.load(f)
    else:
        memory_data = []

    return memory_data

def get_config(company):
    """
    Return configuration JSON files for ChatChain.

    Args:
        company: Customized configuration name under CompanyConfig/

    Returns:
        Tuple containing paths to three configuration JSONs: 
        (config_path, config_phase_path, config_role_path)
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

def data2json(total_data, output_path):
    """
    Convert a DataFrame to JSON and save to a file.

    Args:
        total_data: Pandas DataFrame containing the data.
        output_path: Path to save the JSON file.
    """
    data_dict = total_data.to_dict(orient='records')  # Convert to list of dicts

    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=4)

def run_threaded(start_idx, end_idx, progress_bar, category_data, index, total_data, config_paths, rag_data, memory_index, memory_data, model):
    """
    Function to process a chunk of data in a thread and update results.

    Args:
        start_idx: Starting index of the chunk.
        end_idx: Ending index of the chunk.
        progress_bar: tqdm progress bar instance.
        category_data: DataFrame containing the category-specific data.
        index: Indices corresponding to the category data.
        total_data: The main DataFrame to store results.
        config_paths: Tuple of configuration file paths.
        rag_data: RAG data.
        memory_index: Memory index data.
        memory_data: Memory data.
        model: Model name to use.
    """
    config_path, config_phase_path, config_role_path = config_paths

    for i in range(start_idx, end_idx):
        question = category_data['question'][i]
        #if is NLGraph, the question should add output format
        #question = question + answer_format_dict[category_data['type'][i]]
        library = category_data['search_result'][i]

        # Handle NaN values
        if pd.isna(library):
            library = None

        # Initialize ChatChain
        chat_chain = ChatChain(
            config_path=config_path,
            config_phase_path=config_phase_path,
            config_role_path=config_role_path,
            question=question,
            model_name=model,
            rag_data=rag_data,
            memory_index=memory_index,
            memory_data=memory_data,
            library=library
        )
        chat_chain.make_recruitment()
        chat_chain.execute_chain(lock)

        # Update the main DataFrame with results
        idx = index[i]
        total_data.at[idx, 'result'] = chat_chain.chat_env.env_dict.get("Output")
        total_data.at[idx, 'run'] = chat_chain.chat_env.env_dict.get("run")
        total_data.at[idx, 'code'] = chat_chain.chat_env.env_dict.get("Codes")
        if library is None:
            total_data.at[idx, 'search_result'] = chat_chain.chat_env.env_dict.get("Search_Result")

        print(f"Generated reply for row {idx}: {total_data.at[idx, 'result']}")

        progress_bar.update(1)

def run_all(category_data, index, total_data, config_paths, rag_data, memory_index, memory_data, model, num_threads=1):
    """
    Run data processing in multiple threads and update the main DataFrame.

    Args:
        category_data: DataFrame containing the category-specific data.
        index: Indices corresponding to the category data.
        total_data: The main DataFrame to store results.
        config_paths: Tuple of configuration file paths.
        rag_data: RAG data.
        memory_index: Memory index data.
        memory_data: Memory data.
        model: Model name to use.
        num_threads: Number of threads to use.
    """
    threads = []
    chunk_size = (len(category_data) + num_threads - 1) // num_threads
    global_progress_bar = tqdm(total=len(category_data), desc="Total Progress")

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(category_data))

        thread = threading.Thread(
            target=run_threaded,
            args=(
                start_idx, end_idx, global_progress_bar, category_data, index, 
                total_data, config_paths, rag_data, memory_index, memory_data, model
            )
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_progress_bar.close()

def run_one(question, index, total_data, config_paths, rag_data, memory_index, memory_data, model):
    """
    Process a single question and update the main DataFrame.

    Args:
        question: The question to process.
        index: Index of the question in the main DataFrame.
        total_data: The main DataFrame to store results.
        config_paths: Tuple of configuration file paths.
        rag_data: RAG data.
        memory_index: Memory index data.
        memory_data: Memory data.
        model: Model name to use.
    
    Returns:
        The search result library.
    """
    config_path, config_phase_path, config_role_path = config_paths

    chat_chain = ChatChain(
        config_path=config_path,
        config_phase_path=config_phase_path,
        config_role_path=config_role_path,
        question=question,
        model_name=model,
        rag_data=rag_data,
        memory_index=memory_index,
        memory_data=memory_data,
        library=None
    )
    chat_chain.make_recruitment()
    chat_chain.execute_chain(lock)
    total_data.at[index, 'result'] = chat_chain.chat_env.env_dict.get("Output")
    library = chat_chain.chat_env.env_dict.get("Search_Result")

    return library

def get_categories(data):
    """
    Get all unique categories from the 'type' column in the data.

    Args:
        data: Pandas DataFrame.

    Returns:
        Numpy array of unique categories.
    """
    if 'type' not in data.columns:
        data['type'] = "default"
    return data['type'].unique()

def get_category_data(data, category):
    """
    Get all data and indices for a specific category.

    Args:
        data: Pandas DataFrame.
        category: The category to filter by.

    Returns:
        Tuple containing the filtered DataFrame and its indices.
    """
    filtered_data = data[data['type'] == category]
    return filtered_data.reset_index(drop=True), filtered_data.index

def get_categories_finished(data):
    """
    Get unique categories from data where 'result' is not NaN.

    Args:
        data: Pandas DataFrame.

    Returns:
        Numpy array of unique categories with completed results.
    """
    filtered_df = data[data['result'].notna()]
    return filtered_df['type'].unique()

def merge_dataframes(total_data, finished_data):

    columns_to_merge = ['result', 'run', 'code', 'search_result']

    for col in columns_to_merge:
        if col not in total_data.columns:
            total_data[col] = pd.Series([None] * len(total_data), index=total_data.index)
    
    total_data = total_data.reset_index(drop=True)
    finished_data = finished_data.reset_index(drop=True)
    
    for col in columns_to_merge:
        if col in finished_data.columns:
            total_data[col] = total_data[col].fillna(finished_data[col])
    
    return total_data

def main(args):

    # Load RAG data
    json_files = ["filtered_networkx_reference_edition.json"]
    rag_data = load_rag_data(root, json_filenames=json_files)

    # Load configurations
    config_path, config_phase_path, config_role_path = get_config(args.config)

    # Set model from arguments
    model = args.model_name

    # Load memory data
    memory_index = load_memory_data("GraphTeam/memory/memory_index.json")
    memory_data = load_memory_data("GraphTeam/memory/memory_info.json")

    # Define file paths
    file_path = args.input
    output_path = args.output
    
    global total_data, finished_data
    
    finished_data_copy = pd.DataFrame()

    # Load input data
    total_data = pd.read_json(file_path, orient='records', dtype={'search_result': 'object'})
    total_data['result'] = None
    total_data['run'] = None
    total_data['code'] = None
    total_data['search_result'] = None

    # Get all categories
    categories = get_categories(total_data)

    finished_file_path = Path(output_path)


    if not finished_file_path.exists():
        categories_finished = []
    else:
        finished_data = pd.read_json(output_path)
        total_data['result'] = finished_data.get('result', None).to_list()
        total_data['run'] = finished_data.get('run', None).to_list()
        total_data['code'] = finished_data.get('code', None).to_list()
        if 'search_result' not in finished_data.columns:
            finished_data['search_result'] = None
        total_data['search_result'] = finished_data.get('search_result', None).astype(object).to_list()

        categories_finished = get_categories_finished(finished_data)
        categories = list(set(categories) - set(categories_finished))

    for category in categories:

        print(f"Processing category {category}")

        category_data, category_index = get_category_data(total_data, category)

        run_all(
            category_data, 
            category_index, 
            total_data, 
            (config_path, config_phase_path, config_role_path), 
            rag_data, 
            memory_index, 
            memory_data, 
            model,
            num_threads=args.num_threads
        )
        
        merged_data = merge_dataframes(total_data, finished_data_copy)
        
        # Save results after processing each category
        data2json(merged_data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process benchmark data with ChatChain.")
    
    # Define available model names
    available_models = [
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0125',
        'gpt-3.5-turbo-1106',
        'gpt-4-turbo',
        'gpt-4-turbo-2024-04-09',
        'gpt-4-1106-preview',
        'gpt-4o-mini',
        'gpt-4o-mini-2024-07-18',
        'gpt-4o',
        'gpt-4o-2024-08-06',
        'gpt-4o-2024-05-13',
        'chatgpt-4o-latest'
    ]

    parser.add_argument(
        '--model_name',
        type=str,
        choices=available_models,
        default='gpt-4o-mini',
        help="Specify the model name. Available options: {}".format(', '.join(available_models))
    )

    parser.add_argument(
        '--num_threads',
        type=int,
        default=10,
        help="Number of threads to use for processing (default: 10)"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='Default',
        help="Specify the configuration file to use (default: 'Default')"
    )

    parser.add_argument(
        '--input',
        type=str,
        default='total_benchmark/NLGraph/NLGraph_test.json',
        help="Specify the input file path (default: 'total_benchmark/NLGraph/NLGraph_test.json')"
    )

    parser.add_argument(
        '--output',
        type=str,
        default='total_benchmark/NLGraph/result.json',
        help="Specify the output file path (default: total_benchmark/NLGraph/result.json')"
    )

    args = parser.parse_args()
    main(args)
