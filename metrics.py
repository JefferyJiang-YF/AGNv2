import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict

from sklearn.metrics import classification_report, f1_score
from itertools import chain

from tqdm import tqdm


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
                emissions, _ = self.model(token_ids, segment_ids, tfidf_vectors)

                preds = self.model.crf.decode(emissions)  # 使用 CRF 解码

                # 遍历批次中的每个样本
                for i, length in enumerate((token_ids != 0).sum(1)):
                    # 截取到实际序列长度的标签
                    true_labels_seq = true_labels[i][:length].tolist()
                    pred_labels_seq = preds[i][:length]
                    # print(f"Adding true labels sequence: {true_labels_seq}")  # 调试输出
                    # print(f"Adding predicted labels sequence: {pred_labels_seq}")  # 调试输出
                    y_true.append(true_labels_seq)  # 添加整个序列
                    y_pred.append(pred_labels_seq)  # 添加整个序列

        # 展平 y_true 和 y_pred 列表
        y_true_flat = [label for sublist in y_true for label in sublist]
        y_pred_flat = [label for sublist in y_pred for label in sublist]

        # 输出分类报告
        c_r = classification_report(y_true_flat, y_pred_flat, zero_division=0)
        print(c_r)
        # 计算并返回 F1 得分
        f1 = f1_score(y_true_flat, y_pred_flat, average="micro", zero_division=0)
        return f1, c_r

    def on_epoch_end(self, epoch):
        val_f1, c_r = self.calc_metrics()
        self.history['val_f1'].append(val_f1)
        self.history['c_r'].append(c_r)

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


class NERTestMetrics(torch.nn.Module):
    def __init__(self, model, eval_loader):
        super(NERTestMetrics, self).__init__()
        self.model = model
        self.eval_loader = eval_loader
        self.history = defaultdict(list)

    def calc_metrics(self):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in tqdm(self.eval_loader):
                token_ids, segment_ids, tfidf_vectors, true_labels = data
                emissions, _ = self.model(token_ids, segment_ids, tfidf_vectors)
                preds = self.model.crf.decode(emissions)  # 使用 CRF 解码

                for i, length in enumerate((token_ids != 0).sum(1)):
                    true_labels_seq = true_labels[i][:length].tolist()
                    pred_labels_seq = preds[i][:length]
                    y_true.append(true_labels_seq)
                    y_pred.append(pred_labels_seq)

        y_true_flat = [label for sublist in y_true for label in sublist]
        y_pred_flat = [label for sublist in y_pred for label in sublist]

        c_r = classification_report(y_true_flat, y_pred_flat, zero_division=0)
        print(c_r)

        f1 = f1_score(y_true_flat, y_pred_flat, average="micro")
        return f1, c_r

    def evaluate(self):
        test_f1, c_r = self.calc_metrics()
        self.history['test_f1'].append(test_f1)
        self.history['c_r'].append(c_r)

        print(f"Test set Micro F1: {test_f1:.4f}")
        return test_f1, c_r

