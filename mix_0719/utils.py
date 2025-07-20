import json
from pathlib import Path
import logging

def load_unique_devices_from_file(file_path):
    """
    从 graph_entities.json 文件加载数据，并返回一个包含所有独特设备描述的列表。

    参数:
    - file_path (str or Path): graph_entities.json 文件的路径。

    返回:
    - list: 一个包含所有独特设备描述字符串的列表。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logging.error(f"设备信息文件未找到: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    device_descriptions = set()
    for record in data:
        # 寻找包含'device'信息的实体
        if 'entities' in record:
            for entity in record['entities']:
                if entity.get('type') == 'device':
                    # 将设备描述添加到集合中以保证唯一性
                    device_descriptions.add(entity['name'])
    
    logging.info(f"从 {file_path} 中找到 {len(device_descriptions)} 个独特的设备描述。")
    return list(device_descriptions) 