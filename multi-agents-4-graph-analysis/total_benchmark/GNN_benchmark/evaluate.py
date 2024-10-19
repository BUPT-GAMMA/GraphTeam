from tqdm import tqdm
import threading
import pandas as pd
import re


def compare_code(example_code, student_code):  
    score = 0.0
    #graph_model
    try:
        graph_model = re.findall(r"graph_models=\['(.*?)'\]", example_code)[0]
        student_model = re.findall(r"graph_models=\['(.*?)'\]", student_code)[0]
        if graph_model == student_model:
            score += 0.25
    except:
        pass
    #num_layers
    try:
        num_layers = int(re.findall(r"'num_layers':\s*(\d+)", example_code)[0])
        student_layers = int(re.findall(r"'num_layers':\s*(\d+)", student_code)[0])
        if num_layers == student_layers:
            score += 0.25
    except:
        pass

    #hidden
    try:
        hidden_layers = re.findall(r"'hidden':\s*\[(.*?)\]", example_code)[0]
        hidden_layers = [int(i) for i in hidden_layers.split(',')]
        student_hidden_layers = re.findall(r"'hidden':\s*\[(.*?)\]", student_code)[0]
        student_hidden_layers = [int(i) for i in student_hidden_layers.split(',')]
        equal = all(a == b for a, b in zip(hidden_layers, student_hidden_layers))
        if equal:
            score += 0.25
    except:
        pass

    #dropout
    try:
        dropout_value = float(re.findall(r"'dropout':\s*([\d\.]+)", example_code)[0])
        student_dropout_value = float(re.findall(r"'dropout':\s*([\d\.]+)", student_code)[0])
        if dropout_value == student_dropout_value:
            score += 0.25
    except:
        pass

    return score

def check_threaded(index, results, progress_bar):
    example = data['example_code'][index]
    student_answer = data['code'][index]
    results[index] = compare_code(example, student_answer)        
    progress_bar.update(1)
    
def run_threaded_check(start_idx, end_idx, results, progress_bar):
    for i in range(start_idx, end_idx):
        check_threaded(i, results, progress_bar)
        
def check_answers_by_gpt():
    num_threads = 20
    results = [None] * len(data)
    threads = []

    chunk_size = (len(data) + num_threads - 1) // num_threads

    global_progress_bar = tqdm(total=len(data), desc="Total Progress")

    for i in range(num_threads):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(data))

        thread = threading.Thread(target=run_threaded_check, args=(start_idx, end_idx, results, global_progress_bar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    global_progress_bar.close()
    
    return results
    
def get_score():
    results = check_answers_by_gpt()
    total_score = sum(results) / len(results)
    return total_score


path = 'total_benchmark/GNN_benchmark/result.json'
data = pd.read_json(path)
score = get_score()
print(score)