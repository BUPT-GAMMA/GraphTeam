import json
import re
import pandas as pd
from pathlib import Path
import argparse


CHOICES = ['A', 'B', 'C', 'D', 'E', 'F','G', 'H', 'I', 'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0
    
    
def check(key, truth, predict):
    
    if key in ['cycle', 'connectivity', 'hamilton', 'substructure', 'bipartite']:
        if '###' in predict:
            if 'yes' in truth.lower() and 'yes' in predict.split('###')[-1].lower():
                # correct_samples[key].append(v)
                return True
            elif 'no' in truth.lower() and 'no' in predict.split('###')[-1].lower():
                return True
            return False
        else:
            matches = re.findall(r'\b(yes|no)\b', predict, flags=re.IGNORECASE)
            if matches:
                last_match = matches[-1].lower()
                if last_match == 'yes' and 'yes' in truth.lower():
                    return True
                elif last_match == 'no' and 'no' in truth.lower():
                    return True
            else:
                return False
    elif key in ['flow', 'shortest', 'triangle']:
      
        t_num = extract_last_num(truth)
        p_num = extract_last_num(predict.split('###')[-1])
        if abs(t_num - p_num) < 1e-2:
            return True
        return False
                
    elif key == 'topology':
        
        # elif key == 'topology':
        
        if '###' in predict:
            pre = predict.split('###')[-1].strip(' ')
            truth = truth.split('###')[-1].strip(' ')
            if truth in pre or pre in truth:
                return True
            return False
        else:
            truth = truth.split('###')[-1].split(',')
            for t in truth:
                if t in predict or t.strip(' ') in predict:
                    return True
            return False



def main(args):

    results = {'cycle': 0, 'connectivity': 0, 'hamilton': 0, 'substructure': 0, 'bipartite': 0, 'flow': 0, 'shortest': 0, 'triangle': 0, 'topology': 0}
    data = pd.read_json(args.file_path)
    for i in range(len(data)):
        truth = data['answer'][i]
        predict = data['result'][i]
        task = data['type'][i]
        if check(task, truth, predict):
            results[task] += 1
    print(results)

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--file_path",
        type=str,
        help="Path to file",
        default="total_benchmark/GraphWiz/GraphWiz_test.json",
    )
    args = parser.parse_args()
    main(args)

