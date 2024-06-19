import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

from sklearn.metrics import classification_report, f1_score
from itertools import chain


class ClfMetrics(torch.nn.Module):
    def __init__(self, model, eval_loader, save_path, min_delta=1e-4, patience=10):
        super(ClfMetrics, self).__init__()

        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.save_path = save_path
        self.eval_data = eval_loader  # 直接使用传入的 DataLoader
        self.history = defaultdict(list)
        self.stop_training = False  # 添加标志以支持早停
        self.best = -np.Inf

    def on_train_begin(self):
        self.step = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for token_ids, segment_ids, tfidf_vectors, true_labels in self.eval_data:  # 直接解包每个批次的数据

                # print("Processing a new chunk...")  # 打印开始处理新的数据块

                # 直接使用解包后的数据，不需要额外处理
                preds, attn_weights = self.model(token_ids, segment_ids, tfidf_vectors)

                pred = torch.argmax(preds, dim=1)
                y_true.extend(true_labels.tolist())
                y_pred.extend(pred.tolist())

                # print("Batch size after processing:", token_ids.size(0))

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print("Calculated Accuracy:", acc)
        print("Calculated F1 Score:", f1)
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

    def should_stop(self):
        return self.stop_training

    def max_accuracy(self):
        return max(self.history['val_acc'], default=0)

    def max_f1(self):
        return max(self.history['val_f1'], default=0)


class NERMetrics(torch.nn.Module):
    def __init__(self, model, eval_loader, save_path, min_delta=1e-4, patience=10):
        super(NERMetrics, self).__init__()
        self.model = model
        self.eval_loader = eval_loader
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_op = np.greater
        self.best = -np.Inf
        self.stop_training = False

        self.history = defaultdict(list)

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in self.eval_loader:
                token_ids, segment_ids, tfidf_vectors, true_labels = data
                emissions = self.model(token_ids, segment_ids, tfidf_vectors)
                preds = self.model.crf.decode(emissions)  # Using CRF to decode
                for i, length in enumerate((token_ids != 0).sum(1)):
                    y_true.extend(true_labels[i][:length].tolist())
                    y_pred.extend(preds[i][:length])  # preds are already decoded using CRF

        print(classification_report(list(chain(*y_true)), list(chain(*y_pred))))
        f1 = f1_score(list(chain(*y_true)), list(chain(*y_pred)), average="micro")
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
                self.stop_training = True
                print('Early stopping triggered.')

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print(f'Epoch {self.stopped_epoch + 1:05d}: early stopping.')

    def should_stop(self):
        return self.stop_training

