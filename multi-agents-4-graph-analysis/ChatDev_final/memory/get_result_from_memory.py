import random

def parse_raw_text_to_dict_memory(raw_text):
    """
    将未经结构化的纯文本解析为字典。
    """
    data = {}
    current_field = None

    # 遍历每行文本，寻找关键词并解析字段
    for line in raw_text.splitlines():
        line = line.strip()  # 去除空格
        # 处理 Section ID 和 Section_id
        if line.startswith("id"):
            current_field = "id"  # 统一处理为 Section ID
            data[current_field] = line.replace("id ", "")

    # 将列表内容转换为字符串，便于处理
    for key in data:
        if isinstance(data[key], list):
            data[key] = " ".join(data[key])

    return data

# Step 1: Define a function to search by IDs in detailed_data
def find_detailed_info_by_ids(ids, detailed_data):
    # 根据多个ID查找详细信息
    return [item for item in detailed_data if str(item['id']) in ids]

# Step 3: Define a function to sort by accuracy and randomly select if necessary
def select_by_accuracy(items):
    if not items:
        return None
    # 按 accuracy 降序排序
    sorted_items = sorted(items, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    # 如果多个条目的 accuracy 一样，随机选择一个
    if len(sorted_items) > 1 and sorted_items[0]['accuracy'] == sorted_items[1]['accuracy']:
        return random.choice(sorted_items)
    else:
        return sorted_items[0]