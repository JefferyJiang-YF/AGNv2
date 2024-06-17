import json

from tqdm import tqdm
from datasets import load_dataset


dataset = load_dataset("glue", "sst2")

with open('train.jsonl', 'w') as writer:
    for obj in tqdm(dataset['train']):
        writer.writelines(json.dumps({'text': obj['sentence'], 'label': obj['label']}, ensure_ascii=False) + '\n')


with open('dev.jsonl', 'w') as writer:
    for obj in tqdm(dataset['validation']):
        writer.writelines(json.dumps({'text': obj['sentence'], 'label': obj['label']}, ensure_ascii=False) + '\n')
