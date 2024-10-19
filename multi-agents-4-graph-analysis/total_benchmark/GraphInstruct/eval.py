import re
import networkx as nx

def extract_graph(question):
    # 初始化一个无向图
    if "directed" in question.lower():
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # 将描述字符串按行分割
    lines = question.strip().split('\n')
    
    # 遍历每一行，提取节点和其连接的节点
    for line in lines:
        # 使用正则表达式提取尖括号中的数字
        numbers = re.findall(r'<(\d+)>', line)
        if not numbers:
            continue  # 如果没有找到数字，跳过该行
        node = int(numbers[0])  # 当前节点
        connections = [int(n) for n in numbers[1:]]  # 连接的节点列表
        for conn in connections:
            G.add_edge(node, conn)  # 添加无向边
    return G

def extract_path(path):
    numbers = re.findall(r'\d+', path)
    path = [int(num) for num in numbers]
    return path

def extract_start(question):
    match = re.search(r'Start from node\s*<(\d+)>', question)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("未找到起始节点。请确保描述字符串中包含'开始节点'的信息。")

def dfs_evaluate(question, path):
    graph = extract_graph(question)
    path = extract_path(str(path))
    start = extract_start(question)

    if not path:
        return False

    if start is None:
        start = path[0]

    visited = set()
    stack = [start]
    visited.add(start)
    path_index = 1  # 路径中下一个要访问的节点

    while stack and path_index < len(path):
        current = stack[-1]
        neighbors = list(graph.neighbors(current))
        
        # 过滤已访问的邻居
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        
        if unvisited_neighbors:
            next_node = path[path_index]
            if next_node in unvisited_neighbors:
                stack.append(next_node)
                visited.add(next_node)
                path_index += 1
            else:
                # 下一个节点不是当前节点的未访问邻居，非法DFS路径
                return False
        else:
            stack.pop()

    # 检查是否所有节点都被访问并且路径被完整遍历
    return path_index == len(path) and len(visited) == len(graph)


def hamiltonian_path_evaluate(question, result):
    G = extract_graph(question)
    result = extract_path(str(result))
    if len(set(result)) != G.number_of_nodes() or len(result) != G.number_of_nodes():
        return False

    correct = True
    for i in range(1, len(result)):
        if not G.has_edge(result[i - 1], result[i]):
            correct = False
            break
    return correct

def extract_list(answer):
    numbers = re.findall(r'\d+', answer)
    return [(int(numbers[i]), int(numbers[i+1])) for i in range(0, len(numbers), 2)]

def extract_node_list(answer):
    numbers = re.findall(r'\d+', answer)
    return [int(num) for num in numbers]

def bipartite_evaluate(question, answer, result):
    ans = extract_list(str(answer))
    result = extract_list(str(result))
    if len(result) != len(ans):
        return False

    tmp = question[question.find('Nodes set 1 contains: ') + len('Nodes set 1 contains: '):]
    tmp = tmp[:tmp.find('Nodes set 2 contains: ')]
    node_set_1 = extract_node_list(tmp)
    node_set_1 = set(node_set_1)

    tmp = question[question.find('Nodes set 2 contains: ') + len('Nodes set 2 contains: '):]
    node_set_2 = extract_node_list(tmp)
    node_set_2 = set(node_set_2)

    set1_nodes = set()
    set2_nodes = set()
    for e in result:
        if e[0] in node_set_1 and e[1] in node_set_2:
            pass
        elif e[0] in node_set_2 and e[1] in node_set_1:
            e = (e[1], e[0])
        else:
            return False
            
        if e[0] in set1_nodes:
            return False
        set1_nodes.add(e[0])

        if e[1] in set2_nodes:
            return False
        set2_nodes.add(e[1])
    return True

def extract_float(result):
    match = re.findall(r'\d+\.\d+', result)
    if match:
        #提取最后一个浮点数
        return float(match[-1])
    else:
        match = re.search(r'\d+', result)
        if match:
            return float(match.group())
        else:
            return None


def clustering_coefficient_evaluate(answer, result):
    answer = extract_float(str(answer))
    result = extract_float(str(result))
    if answer is None or result is None:
        return False
    if abs(answer - result) / (answer + 1e-8) < 0.03:
        return True
    return False