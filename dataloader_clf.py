import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import os
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from model import VariationalAutoencoder, Autoencoder

# Seed setting
seed_value = int(os.getenv('RANDOM_SEED', -1))
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


class TextDataset:
    def __init__(self, train_set):
        self.data = train_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 假设 train_set 是一个列表，每个元素是一个字典，包含了 token_ids, segment_ids, tfidf_vector, 和 label_id
        return {
            'token_ids': self.data[idx]['token_ids'],
            'segment_ids': self.data[idx]['segment_ids'],
            'tfidf_vector': self.data[idx]['tfidf_vector'],
            'label_id': self.data[idx]['label_id']
        }


class DataGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.loader = None

    def set_dataset(self, train_set):
        self.dataset = TextDataset(train_set)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        # 确保批次不为空
        if not batch:
            raise ValueError("Received an empty batch. Check your dataset and DataLoader settings.")

        # 准备数据进行填充
        batch_token_ids = [torch.tensor(item['token_ids'], dtype=torch.long) for item in batch]
        batch_segment_ids = [torch.tensor(item['segment_ids'], dtype=torch.long) for item in batch]
        batch_tfidf = [torch.tensor(item['tfidf_vector'], dtype=torch.float) for item in batch]
        batch_labels = [torch.tensor(item['label_id'], dtype=torch.long) for item in batch]

        # 使用 PyTorch 的 pad_sequence 来处理填充
        batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
        batch_segment_ids = pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
        batch_tfidf = torch.stack(batch_tfidf)  # 假设 tfidf 向量已经是合适的张量形式

        batch_labels = torch.stack(batch_labels)  # 假设标签 ID 已经是合适的张量形式

        return [batch_token_ids, batch_segment_ids, batch_tfidf], batch_labels

    def __iter__(self):
        # 确保数据加载器已正确初始化
        if self.loader is None:
            raise RuntimeError("Dataset loader is not initialized. Call 'set_dataset' first.")
        return iter(self.loader)

    def __len__(self):
        # 返回 DataLoader 的长度
        return len(self.loader)

    @property
    def steps_per_epoch(self):
        return len(self.loader)


class CustomDataLoader:
    def __init__(self, tokenizer, ae_latent_dim=128, use_vae=False, batch_size=64, ae_epochs=20):
        self._train_set = []
        self._dev_set = []
        self._test_set = []

        self.use_vae = use_vae
        self.batch_size = batch_size
        self.ae_latent_dim = ae_latent_dim
        self.ae_epochs = ae_epochs
        self.tokenizer = tokenizer
        self.label2idx = {}

        self.tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_features=5000)
        self.autoencoder = None

    def init_autoencoder(self):
        if self.autoencoder is None:
            input_dim = len(self.tfidf.vocabulary_)
            model_class = VariationalAutoencoder if self.use_vae else Autoencoder
            self.autoencoder = model_class(input_dim=input_dim, latent_dim=self.ae_latent_dim).to('cuda' if torch.cuda.is_available() else 'cpu')

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
        return len(self.label2idx)

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

    def prepare_tfidf(self, data, is_train=False):

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

        if is_train:
            self.init_autoencoder()
            self.autoencoder.fit(X)

        X = self.autoencoder.encode(X)

        print('>>> Final X shape:', X.shape)

        assert len(X) == len(data)
        for x, obj in zip(X, data):
            obj['tfidf_vector'] = x.tolist()
        return data

    def _read_data(self, fpath, build_vocab=False):
        data = []
        tfidf_corpus = []
        with open(fpath, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line)
                obj['text'] = clean_str(obj['text'])  # Assuming clean_str is defined elsewhere
                tfidf_corpus.append(obj['text'])

                # Tokenize text for every line regardless of the condition
                encoded = self.tokenizer(
                    obj['text'],
                    add_special_tokens=True,
                    return_token_type_ids=True,
                    return_attention_mask=False,
                    return_tensors=None
                )

                # Update label index if necessary
                if build_vocab and obj['label'] not in self.label2idx:
                    self.label2idx[obj['label']] = len(self.label2idx)

                # Collect token_ids and segment_ids from encoded
                token_ids = encoded['input_ids']
                segment_ids = encoded['token_type_ids']
                data.append({
                    'raw_text': obj['text'],
                    'token_ids': token_ids,
                    'segment_ids': segment_ids,
                    'label_id': self.label2idx[obj['label']]
                })

        # Building or updating the TF-IDF model if required
        if build_vocab:
            self.tfidf.fit(tfidf_corpus)
            # print(build_vocab)  # Debugging statement

        # Assuming prepare_tfidf is a method that processes data and applies TF-IDF transformations
        return self.prepare_tfidf(data, is_train=build_vocab)



class NewDataset(Dataset):
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