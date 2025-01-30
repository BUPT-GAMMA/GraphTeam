from tqdm import tqdm
import openai
from tenacity import retry, wait_exponential, stop_after_attempt
import threading
import json
import pandas as pd
import eval

openai.api_key = ''
openai.base_url = ''
model_use = 'gpt-4o-mini'
        
question_prompt = """
        Question:
        {question}

        Reference Answer:
        {reference_answer}

        Student Answer:
        {student_answer}
        """

@retry(wait=wait_exponential(multiplier=1, max=360), stop=stop_after_attempt(50))
def gpt_check_answer(question, model_name=model_use):
    # 调用 GPT-3.5-turbo-1106 API 生成回答

    # 3.5 turbo: gpt-3.5-turbo-1106
    # 4 turbo: gpt-4-1106-preview
    completion = openai.chat.completions.create(
        model=model_name,  # 使用 GPT-3.5-turbo-1106 模型
        messages=[
            {"role": "system", "content": "You are a teacher who is marking test papers. I will give you a question, reference answer(s) and the student’s answer. You need to understand the question, and based on the reference answer(s), decide if the student’s answer is correct. Please note that there may be more than one reference answer, and the student’s answer only needs to be completely consistent with any one of them to be considered correct."},
            {"role": "system", "content": "You need to give me 1 or 0 based on the student's answer.\
                                           You need to response in json format.\
                                           Your check_result_score should belong to [0, 1]. The max check_result_score is 1.\
                                           reason(str): your reason for the check_result.\
                                           check_result_score(float): 1 or 0"},
            {"role": "user", "content": question},
        ],
        
        temperature = 0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content.strip())

def check_threaded(index, results, progress_bar):
    #如果太长，只保留前1000个字符
    type = data['type'][index]
    question = data['question'][index]
    reference_answer = data['answer'][index][:1000]
    student_answer = str(data['result'][index])[:1000] if len(str(data['result'][index])) > 1000 else data['result'][index]
    if type == 'DFS':
        results[index] = eval.dfs_evaluate(question, student_answer)
    elif type == 'hamiltonian_path':
        results[index] = eval.hamiltonian_path_evaluate(question, student_answer)
    elif type == 'bipartite':
        results[index] = eval.bipartite_evaluate(question, reference_answer, student_answer)
    elif type == 'clustering_coefficient':
        results[index] = eval.clustering_coefficient_evaluate(reference_answer, student_answer)
    else:
        question = question_prompt.format(question=question, reference_answer=reference_answer, student_answer=student_answer)
    results[index] = gpt_check_answer(question)
        
    progress_bar.update(1)
    
def run_threaded_check(start_idx, end_idx, results, progress_bar):
    for i in range(start_idx, end_idx):
        check_threaded(i, results, progress_bar)
        
def check_answers_by_gpt():
    num_threads = 30
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
    scores = []
    results = check_answers_by_gpt()
    total_score = 0.0
    #保存分数和理由到文件
    for i in range(len(results)):
        json_data = results[i]
        if json_data['check_result_score'] >= 1.0:
            score = 1
        elif  json_data['check_result_score'] < 0:
            score = 0
        else:
            score = float(json_data['check_result_score'])
        scores.append(score)
    return scores

data = pd.read_json('G:/multi-agents-4-graph-analysis/total_benchmark/GraphInstruct/benchmark.json')
data = data.dropna(subset=['result']).reset_index(drop=True)
scores = get_score()
print("Final Score: ", sum(scores) / len(scores))
#打印各类别的分数
for type in data['type'].unique():
    type_scores = [scores[i] for i in range(len(scores)) if data['type'][i] == type]
    print(type, "Score: ", sum(type_scores) / len(type_scores))
