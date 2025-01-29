import re
import pandas as pd
import numpy as np
import networkx as nx

def evaluate_connectivity(result, answer):
    """
    Evaluate the connectivity task
    Args:
        result: string, a prediction of connectivity
        answer: string, the correct answer of connectivity
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    accuracy = 1 if result.lower().strip() == answer.lower().strip() else 0
    return accuracy

def evaluate_cycle(result, answer):
    """
    Evaluate the cycle task
    Args:
        result: string, a prediction of cycle
        answer: string, the correct answer of cycle
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    accuracy = 1 if result.lower().strip() == answer.lower().strip() else 0
    return accuracy

def evaluate_flow(result, answer, question):
    """
    Evaluate the flow task
    Args:
        result: string, a prediction of flow
        answer: string, the correct answer of flow
        question: string, the question of flow
    Returns:
        flag1: 1 if the prediction is correct, 0 otherwise(guess means if the result has the correct format)
        flag2: 1 if the prediction is correct, 0 otherwise(means if the result has the correct value)
    """
    match = re.search(r"Q: What is the maximum flow from node (\d+) to node (\d+)?", question)
    if match:
        q = [int(match.group(1)), int(match.group(2))]
    else:
        return 0, 0

    answer_match = re.search(r"The maximum flow from node (\d+) to node (\d+) is (\d+).", answer)
    if answer_match:
        answer = int(answer_match.group(3))
    else:
        return 0, 0
        
    mode_str = "the maximum flow from node "+str(q[0])+" to node " + str(q[1])
    pos = result.find(mode_str)
    if pos == -1:
        mode_str = "maximum flow from node "+str(q[0])+" to node " + str(q[1])
        pos = result.find(mode_str)
        if pos == -1:
            return 0, 0
    flag1, flag2 = 1, 1
    pos = pos + len(mode_str) + 1
    i = pos
    while i < len(result) and not (result[i] >= '0' and result[i] <='9'):
        i+=1
    num = 0
    while i < len(result) and result[i] >= '0' and result[i] <='9':
        num = num*10 + int(result[i])
        i+=1
    
    return flag1, flag2

def extract_embedding_gnn(answer):
    """
    Extract the embedding from the answer of gnn
    Args:
        answer: string, the answer of gnn
    Returns:
        embedding: list, the embedding extracted from the answer
    """
    matches = re.findall(r'node (\d+): \[([^\]]+)\]', answer)
    data = []
    for match in matches:
        node, values = match
        values = list(map(int, values.split(',')))
        data.append(values)
    return data
    

def evaluate_gnn(result, answer, question):
    """
    Evaluate the gnn task
    Args:
        result: string, a prediction of gnn
        answer: string, the correct answer of gnn
        question: string, the question of gnn
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    match = re.search(r"In an undirected graph, the nodes are numbered from (\d+) to (\d+),", question)
    if match:
        n = int(match.group(2)) - int(match.group(1)) + 1
    else:
        return 0
    answer = extract_embedding_gnn(answer)
    
    mode_string = "node " + str(0)
    pos = result.rfind(mode_string)
    if pos == -1:
        return 0
    result = list(re.findall("\d+", result[pos:]))
    flag = 1
    for i in range(n):
        for j in range(len(answer[i])):
            if abs(int(result[i*(len(answer[i])+1)+j+1]) - int(answer[i][j])) > 0.01:
                return 0
    return flag

def extract_graph_hamilton(question):
    """
    Extract the graph from the question of hamilton
    Args:
        question: string, the question of hamilton
    Returns:
        G: nx.Graph, the graph extracted from the question
    """
    match = re.search(r'The nodes are numbered from (\d+) to (\d+)', question)
    if match:
        n = int(match.group(2)) - int(match.group(1)) + 1
    else:
        return None
    node_pairs = re.findall(r'\((\d+),(\d+)\)', question)
    node_pairs = [(int(i), int(j)) for i, j in node_pairs]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(node_pairs)
    
    return G    

def check_hamilton(solution, G):
    n = G.number_of_nodes()
    if len(solution) != n:
        return 0
    for i in range(len(solution)-1):
        if not G.has_edge(solution[i], solution[i+1]):
            #print(solution[i], solution[i+1])
            return 0
    for i in range(len(solution)-1):
        for j in range(i+1, len(solution)-1):
            if solution[i] == solution[j]:
                return 0
    return 1

def process_ans_hamilton(ans, pos, G):
    num, flag = 0, 0
    solution = []
    n = G.number_of_nodes()
    for i in range(pos, len(ans)):
        if ans[i] >= '0' and ans[i] <='9':
            num = num*10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if len(solution) == n:
                    break
                flag = 0
            num = 0     
    if flag == 1 and ans[-1] >= '0' and ans[-1] <='9':
        solution.append(num)
    return solution

def evaluate_hamilton(result, answer, question):
    """
    Evaluate the hamilton task
    Args:
        result: string, a prediction of hamilton
        answer: string, the correct answer of hamilton
        question: string, the question of hamilton
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    G = extract_graph_hamilton(question)
    if G is None:
        return 0
    pos = result.find("the path can be")
    pos2 = result.find("no ")
    if pos2 == -1:
        pos2 = 10000000
    if pos2 < pos:
        return 0
    if pos == -1:
        return 0
    solution = process_ans_hamilton(result, pos, G)
    flag  = check_hamilton(solution, G)
    return flag

def extract_graph_matching(question):
    """
    Extract the graph from the question of matching
    Args:
        question: string, the question of matching
    Returns:
        G: nx.Graph, the graph extracted from the question
    """
    #node
    match = re.search(r'There are (\d+) job applicants numbered from (\d+) to (\d+), and (\d+) jobs numbered from (\d+) to (\d+). ', question)
    n1 = int(match.group(1))
    n2 = int(match.group(4))
    #edge
    node_pairs = re.findall(r'Applicant (\d+) is interested in job (\d+)', question)
    node_pairs = [(int(i), int(j) + n1) for i, j in node_pairs]
    
    G = nx.Graph()
    G.add_nodes_from(range(n1), bipartite = 0)
    G.add_nodes_from(range(n1, n1+n2), bipartite = 1)
    G.add_edges_from(node_pairs)

    return G, n1

def evaluate_matching(result, answer, question):
    """
    Evaluate the matching task
    Args:
        result: string, a prediction of matching
        answer: string, the correct answer of matching
        question: string, the question of matching
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    G, n1 = extract_graph_matching(question)
    if G is None:
        return 0
    match = re.search(r'(\d+) applicants can find the job they are interested in.', answer)
    if match:
        std = int(match.group(1))
    else:
        return 0
    
    i = result.find("applicant")
    j = result.find("job")
    tag = 0
    if j < i:
        tag = 1
        i = j
    if i == -1:
        return 0
    solution = []
    pos = result.find("this way")
    i = max(i, result.find("make the following assignments"))
    if pos == -1:
        pos = len(result)
    while(i < pos):
        while i < pos and not (result[i] >= '0' and result[i] <='9'):
            i+=1
        if i == pos:
            break
        num = 0
        while i < pos and result[i] >= '0' and result[i] <='9':
            num = num*10 + int(result[i])
            i+=1
        while i < pos and not (result[i] >= '0' and result[i] <='9'):
            i+=1
        if i == pos:
            break
        num2 = 0
        while i < pos and result[i] >= '0' and result[i] <='9':
            num2 = num2*10 + int(result[i])
            i+=1
        if tag:
            solution.append((num2, num+n1))
        else:
            solution.append((num, num2+n1))
    cnt = 0
    for edg in solution:
        if G.has_edge(edg[0], edg[1]):
            cnt += 1
    for i in range(len(solution)):
        for j in range(i-1):
            if solution[i][0] == solution[j][0] or solution[i][1] == solution[j][1]:
                return 0
    if cnt != std:
        return 0
    return 1

def extract_graph_shortest_path(question):
    match = re.search(r'the nodes are numbered from (\d+) to (\d+)', question)
    if match:
        n = int(match.group(2)) - int(match.group(1)) + 1
    else:
        return None
    node_pairs = re.findall(r'node (\d+) and node (\d+) with weight (\d+)', question)
    node_pairs = [(int(i), int(j), int(w)) for i, j, w in node_pairs]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(node_pairs)
    match = re.search(r'Give the shortest path from node (\d+) to node (\d+)', question)
    if match:
        q = [int(match.group(1)), int(match.group(2))]
    else:
        return None
    return G, q

def evaluate_shortest_path(result, question):
    """
    Evaluate the shortest path task
    Args:
        result: string, a prediction of shortest path
        question: string, the question of shortest path
    Returns:
        flag1: 1 if the prediction is correct, 0 otherwise check for path_length 
        flag2: 1 if the prediction is correct, 0 otherwise check for total weight
    """
    G, q = extract_graph_shortest_path(question)
    entity = "node"
    mode_str = "the shortest path from "+entity+' '+str(q[0])+" to "+ entity + ' ' + str(q[1])
    pos = result.find(mode_str)
    if pos == -1:
        return 0, 0
    pos = pos + len(mode_str) + 1
    num, flag = 0, 0
    solution = []
    for i in range(pos, len(result)):
        if result[i] >= '0' and result[i] <='9':
            num = num*10 + int(result[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if num == q[1]:
                    break
                flag = 0
            num = 0
    length = 0
    flag1, flag2 = 1, 1
    for i in range(len(solution)-1):
        if not G.has_edge(solution[i], solution[i+1]):
            flag1 = 0
            break
        length += G[solution[i]][solution[i+1]]["weight"]
    shortest = nx.shortest_path_length(G, source=q[0], target=q[1], weight="weight")
    if length != shortest:
        flag1 = 0
    pos1 = result.rfind("total length")
    pos2 = result.rfind("total weight")
    if pos1 == -1 and pos2 == -1:
        return flag1, 0
    pos = max(pos1, pos2)
    i = pos
    while i < len(result) and not (result[i] >= '0' and result[i] <='9'):
        i+=1
    num = 0
    while i < len(result) and result[i] >= '0' and result[i] <='9':
        num = num*10 + int(result[i])
        i+=1
    if num != shortest:
        flag2 = 0
    return flag1, flag2

def extract_graph_topology(question):
    match = re.search(r'with (\d+) nodes numbered from (\d+) to (\d+)', question)
    if match:
        n = int(match.group(1))
    else:
        return None
    node_pairs = re.findall(r'node (\d+) should be visited before node (\d+)', question)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, j in node_pairs:
        G.add_edge(int(i), int(j))
    return G

def check_topology(solution, G):
    try:
        n = G.number_of_nodes()
        deg = [0] * n
        for i in range(n):
            deg[i] = G.in_degree[i]
        for node in solution:
            if deg[node] > 0:
                return 0
            for neighbor in list(G[node]):
                deg[neighbor] -= 1
        for i in range(n):
            if deg[i] != 0:
                return 0
        return 1
    except:
        return 0
def process_ans_topology(ans, pos, G):
    num, flag = 0, 0
    solution = []
    n = G.number_of_nodes()
    for i in range(pos, len(ans)):
        if ans[i] >= '0' and ans[i] <='9':
            num = num*10 + int(ans[i])
            flag = 1
        else:
            if flag == 1:
                solution.append(num)
                if len(solution) == n:
                    break
                flag = 0
            num = 0
    return solution

def evaluate_topology(result, question):
    """
    Evaluate the topology task
    Args:
        result: string, a prediction of topology
        answer: string, the correct answer of topology
    Returns:
        flag: 1 if the prediction is correct, 0 otherwise
    """
    G = extract_graph_topology(question)
    pos = result.find("solution")
    if pos == -1:
        pos = max(result.find("yes"), result.find("in the following order"))
    if pos == -1:
        return 0
    solution = process_ans_topology(result, pos, G)
    flag1  = check_topology(solution, G)
    solution = process_ans_topology(result, 0, G)

    flag2  = check_topology(solution, G)
    return (flag1 or flag2)

def evaluate(data):
    total_score = {'connectivity': 0, 'cycle': 0, 'flow': {'score1': 0, 'score2': 0}, 'GNN': 0, 'hamilton': 0, 'matching': 0, 'shortest_path': {'score1': 0, 'score2': 0}, 'topology': 0}
    for i in range(len(data)):
        question = data['question'][i]
        answer = data['answer'][i]
        result = str(data['result'][i]).lower()
        task = data['type'][i]
        if task == 'connectivity':
            score = evaluate_connectivity(result, answer)
            total_score['connectivity'] += score
        elif task == 'cycle':
            score = evaluate_cycle(result, answer)
            total_score['cycle'] += score
        elif task == 'flow':
            score1, score2 = evaluate_flow(result, answer, question)
            total_score['flow']['score1'] += score1
            total_score['flow']['score2'] += score2
        elif task == 'GNN':
            score = evaluate_gnn(result, answer, question)
            total_score['GNN'] += score
        elif task == 'hamilton':
            score = evaluate_hamilton(result, answer, question)
            total_score['hamilton'] += score
            if score == 0:
                print(question)
                print(result)
                print(answer)
        elif task == 'matching':
            score = evaluate_matching(result, answer, question)
            total_score['matching'] += score
        elif task == 'shortest_path':
            score1, score2 = evaluate_shortest_path(result, question)
            total_score['shortest_path']['score1'] += score1
            total_score['shortest_path']['score2'] += score2
        elif task == 'topology':
            score = evaluate_topology(result, question)
            total_score['topology'] += score
    return total_score

def show_score(score):
    print(f"Connectivity: {score['connectivity']} / {len(data[data['type'] == 'connectivity'])}")
    print(f"Cycle: {score['cycle']} / {len(data[data['type'] == 'cycle'])}")
    print(f"Flow: \n    score1: {score['flow']['score1']} / {len(data[data['type'] == 'flow'])} \n    score2: {score['flow']['score2']} / {len(data[data['type'] == 'flow'])}")
    print(f"GNN: {score['GNN']} / {len(data[data['type'] == 'GNN'])}")
    print(f"Hamilton: {score['hamilton']} / {len(data[data['type'] == 'hamilton'])}")
    print(f"Matching: {score['matching']} / {len(data[data['type'] == 'matching'])}")
    print(f"Shortest Path: \n   score1: {score['shortest_path']['score1']} / {len(data[data['type'] == 'shortest_path'])} \n   score2: {score['shortest_path']['score2']} / {len(data[data['type'] == 'shortest_path'])}")
    print(f"Topology: {score['topology']} / {len(data[data['type'] == 'topology'])}")

#读取csv文件
data = pd.read_json('total_benchmark/NLGraph/result.json')
score = evaluate(data)
show_score(score)
total_score = 0
#计算总分
for key in score:
    if type(score[key]) == int:
        total_score += score[key]
    else:
        min_score = [score[key][key2] for key2 in score[key]]
        total_score += min(min_score)
        
print(f"Total Score: {total_score} / {len(data)}")
