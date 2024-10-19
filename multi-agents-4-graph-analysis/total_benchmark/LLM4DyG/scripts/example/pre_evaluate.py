#读取LLM4DYG_5_5_0.3.csv文件，将其中的answer列的内容分别为每个题目都生成一个json文件，文件名为answer_{model}.json，其中model为gpt-4o-mini，文件内容为answer列的内容，并放到对应的文件夹中
import os
import json
import pandas as pd
import numpy as np
data = pd.read_json('G:/multi-agents-4-graph-analysis/total_benchmark/LLM4DyG/benchmark/LLM4DYG_5_5_0.3.json')

model = 'gpt-4o-mini'
for i in range(len(data)):
    folder_path = 'G:/multi-agents-4-graph-analysis/total_benchmark/' + data.iloc[i]['file_path']
    answer = data.iloc[i]['result']
    print(answer)

    if type(answer) is not str:
        answer = str(answer)
    if answer.lower().find('answer:') == -1 and answer.lower().find('answer :') == -1:
        answer = "Answer: " + answer

    answer_path = os.path.join(folder_path, f"answer_{model}.json")
    with open(answer_path, 'w') as f:
        json.dump({'content': answer}, f)

print('done')