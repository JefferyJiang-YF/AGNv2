import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from collections import defaultdict
from seqeval.metrics import f1_score as ner_f1_score, classification_report as ner_classification_report


class ClfMetrics(torch.nn.Module):
    def __init__(self, model, batch_size, eval_data, save_path, min_delta=1e-4, patience=10):
        super(ClfMetrics, self).__init__()

        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.save_path = save_path
        self.batch_size = batch_size
        self.eval_data = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
        self.history = defaultdict(list)
        self.stop_training = False  # 添加标志以支持早停

    def on_train_begin(self):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for chunk in self.eval_data:
                token_ids = pad_sequence([obj['token_ids'] for obj in chunk], batch_first=True, padding_value=0)
                segment_ids = pad_sequence([obj['segment_ids'] for obj in chunk], batch_first=True, padding_value=0)
                tfidf_vectors = torch.stack([obj['tfidf_vector'] for obj in chunk])
                true_labels = torch.tensor([obj['label_id'] for obj in chunk])

                pred = self.model(token_ids, segment_ids, tfidf_vectors)
                pred = torch.argmax(pred, dim=1)
                y_true.extend(true_labels.tolist())
                y_pred.extend(pred.tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        return f1, acc

    def on_epoch_end(self, epoch):
        val_f1, val_acc = self.calc_metrics()
        self.history['val_acc'].append(val_acc)
        self.history['val_f1'].append(val_f1)
        print(f"- val_acc: {val_acc} - val_f1: {val_f1}")

        if self.monitor_op(val_f1 - self.min_delta, self.best):
            self.best = val_f1
            self.wait = 0
            print(f'New best model, saving model to {self.save_path}...')
            torch.save(self.model.state_dict(), self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True  # 设置停止训练标志
                print('Early stopping triggered.')

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print(f'Epoch {self.stopped_epoch + 1:05d}: early stopping.')


class NERMetrics(torch.nn.Module):
    def __init__(self, model, batch_size, eval_data, save_path, min_delta=1e-4, patience=10):
        super(NERMetrics, self).__init__()
        self.model = model
        self.save_path = save_path
        self.batch_size = batch_size
        self.eval_data = DataLoader(eval_data, batch_size=self.batch_size, shuffle=False)
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.history = defaultdict(list)
        self.stop_training = False  # 添加标志以支持早停

    def on_train_begin(self):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def decode(self, tag_ids):
        """Decodes tag indexes to formatted tags."""
        tag_vocab = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC', 5: 'B-ORG',
                     6: 'I-ORG'}  # Example tag mapping
        tags = [tag_vocab.get(tag_id, 'O') for tag_id in tag_ids]
        return tags

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for chunk in self.eval_data:
                inputs = {key: pad_sequence([obj[key] for obj in chunk], batch_first=True, padding_value=0) for key in
                          ['token_ids', 'segment_ids']}
                labels = [obj['label_ids'] for obj in chunk]

                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs, dim=-1)

                y_true.extend([self.decode(label) for label in labels])
                y_pred.extend([self.decode(prediction) for prediction in predictions])

        print(ner_classification_report(y_true, y_pred))
        f1 = ner_f1_score(y_true, y_pred)
        return f1

    def on_epoch_end(self, epoch):
        val_f1 = self.calc_metrics()
        self.history['val_f1'].append(val_f1)
        print(f"- val_f1: {val_f1}")

        if self.monitor_op(val_f1 - self.min_delta, self.best):
            self.best = val_f1
            self.wait = 0
            print(f'New best model, saving model to {self.save_path}...')
            torch.save(self.model.state_dict(), self.save_path)
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True  # 设置停止训练标志
                print('Early stopping triggered.')

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print(f'Epoch {self.stopped_epoch + 1:05d}: early stopping.')