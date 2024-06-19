import numpy as np
import random
import os
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

from model import VariationalAutoencoder, Autoencoder

# Seed setting
seed_value = int(42)
if seed_value != -1:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[\n\t]", " ", string)
    string = re.sub(r" +", " ", string)
    string = string.strip().lower()
    return string


class CustomDataLoader:
    def __init__(self, tokenizer, max_len=512, ae_latent_dim=2048, use_vae=False, batch_size=64, ae_epochs=20):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.max_len = max_len
        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.tokenizer = tokenizer
        self.label2idx = {}
        self._label_size = None

        self.tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_features=5000)
        self.autoencoder = None

    def init_autoencoder(self):
        if self.autoencoder is None:
            input_dim = len(self.tfidf.vocabulary_)
            model_class = VariationalAutoencoder if self.use_vae else Autoencoder
            self.autoencoder = model_class(input_dim=input_dim, latent_dim=self.ae_latent_dim).to(
                'cuda' if torch.cuda.is_available() else 'cpu')

    def save_vocab(self, save_path):
        with open(save_path, 'wb') as writer:
            pickle.dump({'label2idx': self.label2idx}, writer)

    def load_vocab(self, save_path):
        with open(save_path, 'rb') as reader:
            obj = pickle.load(reader)
            self.label2idx = obj.get('label2idx', {})

    def save_autoencoder(self, save_path):
        torch.save(self.autoencoder.state_dict(), save_path)

    def load_autoencoder(self, save_path):
        self.init_autoencoder()
        self.autoencoder.load_state_dict(torch.load(save_path))

    def set_train(self, train_path):
        self._train_set = self._read_data(train_path, build_vocab=True)  # True

    def set_dev(self, dev_path):
        self._dev_set = self._read_data(dev_path)

    def set_test(self, test_path):
        self._test_set = self._read_data(test_path)

    @property
    def train_set(self):
        return self._train_set

    @property
    def dev_set(self):
        return self._dev_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def label_size(self):
        print('>>> label2idx:', self._label_size)
        return self._label_size

    def save_dataset(self, setname, fpath):
        dataset = getattr(self, f'_{setname}_set', None)
        if dataset is None:
            raise ValueError(f'Not supported set {setname}')
        with open(fpath, 'w') as writer:
            for data in dataset:
                writer.write(json.dumps(data, ensure_ascii=False) + "\n")

    def load_dataset(self, setname, fpath):
        if setname not in ['train', 'dev', 'test']:
            raise ValueError(f'Not supported set {setname}')
        dataset = []
        with open(fpath, 'r') as reader:
            for line in reader:
                dataset.append(json.loads(line.strip()))
        setattr(self, f'_{setname}_set', dataset)

    def prepare_tfidf(self, data, build_vocab=False):

        """

        Prepare TF-IDF vectors for the given data

        """

        # 这里为啥要丢弃128

        if self.use_vae:
            print("batch alignment...")
            print("previous data size:", len(data))

            original_size = len(data)
            truncated_size = len(data) // self.batch_size * self.batch_size
            data = data[:truncated_size]
            print(f"Batch alignment: from {original_size} to {truncated_size}")

        X = self.tfidf.transform([obj['raw_text'] for obj in data]).todense()

        # Convert numpy matrix to PyTorch tensor
        X = torch.tensor(X, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

        print('>>>tf idf vector shape:', X.shape)

        if build_vocab:
            self.init_autoencoder()
            self.autoencoder.fit(X, ae_epochs=self.ae_epochs)

        X = self.autoencoder.encode(X)

        print('>>> Final X shape:', X.shape)

        assert len(X) == len(data)
        for x, obj in zip(X, data):
            obj['tfidf_vector'] = x.tolist()
        return data

    def _read_data(self, fpath, build_vocab=False):
        data = []
        tfidf_corpus = []
        all_label_set = set()

        # 创建或加载tokenizer

        with open(fpath, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line)
                raw_text = ' '.join(obj['tokens'])
                tfidf_corpus.append(raw_text)

                first, last = None, None
                token_ids, tag_ids = [], []
                for i, (tag, token) in enumerate(zip(obj['ner_tags'], obj['tokens'])):
                    all_label_set.add(tag)

                    # 使用tokenizer对单个token进行编码
                    tok = self.tokenizer(
                        token,
                        add_special_tokens=True,  # 添加特殊tokens，如[CLS], [SEP]
                        return_tensors=None
                    )

                    if i == 0:
                        first, last = tok['input_ids'][0], tok['input_ids'][-1]

                    # 将编码后的token ID添加到token_ids列表中
                    token_ids.extend(tok['input_ids'][1:-1])
                    # 重复对应次数的标签ID添加到tag_ids列表中
                    tag_ids.extend([tag] * len(tok['input_ids'][1:-1]))

                # 调整token_ids和tag_ids列表，包括首尾特殊token
                token_ids = [first] + token_ids[:self.max_len - 2] + [last]
                tag_ids = [0] + tag_ids[:self.max_len - 2] + [0]

                assert len(token_ids) == len(tag_ids)

                data.append({
                    'raw_text': raw_text,
                    'token_ids': token_ids,
                    'segment_ids': [0] * len(token_ids),
                    'label_id': tag_ids
                })

        # 根据需要构建或更新TF-IDF模型
        if build_vocab:
            print('fit tfidf...')
            self.tfidf.fit(tfidf_corpus)
            self._label_size = len(all_label_set)

        def writeFiles(data):
            # For test only ====================================
            with open('Test_loadData.txt', 'w', encoding='utf-8') as file:
                for item in data:
                    item_str = json.dumps(item, ensure_ascii=False)
                    file.write(item_str + '\n')

        data = self.prepare_tfidf(data, build_vocab=build_vocab)

        # writeFiles(data)

        # 处理并应用TF-IDF转换
        return data


class NERDataset(Dataset):
    def __init__(self, train_set):
        self.data = train_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 确保返回的是 PyTorch 张量
        return {
            'token_ids': torch.tensor(self.data[idx]['token_ids'], dtype=torch.long),
            'segment_ids': torch.tensor(self.data[idx]['segment_ids'], dtype=torch.long),
            'tfidf_vector': torch.tensor(self.data[idx]['tfidf_vector'], dtype=torch.float),
            'label_id': torch.tensor(self.data[idx]['label_id'], dtype=torch.long)
        }



    # ========================================================
