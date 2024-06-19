import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 加载数据集
dataset = load_dataset("eriktks/conll2003")

# 指定数据存储的基本路径
base_path = 'examples/ner'

# 检查目录是否存在，如果不存在则创建
os.makedirs(base_path, exist_ok=True)

# 定义一个函数来写入数据
def write_to_json(file_path, data):
    with open(file_path, 'w') as writer:
        for obj in tqdm(data):
            writer.writelines(json.dumps({
                'id': obj['id'],
                'tokens': obj['tokens'],
                "pos_tags": obj["pos_tags"],
                "chunk_tags": obj["chunk_tags"],
                "ner_tags": obj["ner_tags"]
            }, ensure_ascii=False) + '\n')

# 写入训练数据
write_to_json(os.path.join(base_path, 'train.json'), dataset['train'])

# 写入验证数据
write_to_json(os.path.join(base_path, 'val.json'), dataset['validation'])

# 写入测试数据
write_to_json(os.path.join(base_path, 'test.json'), dataset['test'])