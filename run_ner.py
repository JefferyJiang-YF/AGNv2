import os
import sys
import json
import random
from pprint import pprint
from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
# # 假设这些自定义模块已经被正确实现并可以导入
from dataloader_ner import CustomDataLoader, NERDataset
from model import AGNModel
from metrics import NERMetrics, NERTestMetrics
from torch.optim.lr_scheduler import StepLR
from torch.optim import AdamW
import logging
from datetime import datetime


# 设置随机种子以确保可重复性

def setup_logging(exp_name: str):
    # 获取当前日期和时间
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 创建日志文件名称
    log_filename = f"log/{exp_name}_{date_time}.txt"

    # 创建一个日志记录器
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建文件处理器，写入日志文件
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)

    return logger


def log_experiment(logger, epoch, config, average_loss, metrics_callback):
    # 记录日志信息
    micro_f1 = max(metrics_callback.history['val_f1'])
    # c_r = str(metrics_callback.history['c_r']) # 这里在存在0值的时候会报BUG #TODO
    logger.debug(f"Epoch {epoch}/{config['epochs']}, Average Loss: {average_loss:.4f}, Micro F1: {micro_f1}")
    # formatted_c_r = ' '.join(c_r.split())
    # logger.info(f"Completed evaluation on val set with classification report: \n{formatted_c_r}")

def collate_fn(batch):
    batch_token_ids = [item['token_ids'] for item in batch]
    batch_segment_ids = [item['segment_ids'] for item in batch]
    batch_tfidf = [item['tfidf_vector'] for item in batch]
    batch_labels = [item['label_id'] for item in batch]

    # 对序列进行填充
    batch_token_ids = pad_sequence(batch_token_ids, batch_first=True, padding_value=0)
    batch_segment_ids = pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
    batch_tfidf = torch.stack(batch_tfidf)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0)

    # 不封装成列表，直接返回多个张量
    return batch_token_ids, batch_segment_ids, batch_tfidf, batch_labels


def load_model_weights(model, weights_path):
    model.load_state_dict(torch.load(weights_path,
                                     map_location=torch.device \
                                         ('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()


def setup_seed(seed=42):
    """设置随机种子以及其他相关配置以确保代码的可复现性。"""
    # 设定Python环境的哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设定Python内建的随机种子
    random.seed(seed)

    # 设定Numpy的随机种子
    np.random.seed(seed)

    # 设定PyTorch的随机种子
    torch.manual_seed(seed)

    # 如果使用CUDA
    if torch.cuda.is_available():
        # 设定所有CUDA设备的随机种子
        torch.cuda.manual_seed_all(seed)
        # 确保所有操作都在CUDA上尽可能地确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 限制PyTorch使用的线程数量
    torch.set_num_threads(1)

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42
    setup_seed(seed)

    if not os.path.exists('log'):
        os.makedirs('log')

    # 读取配置文件
    config_file = 'examples/ner/conll03.json'
    with open(config_file, "r") as reader:
        config = json.load(reader)
    config['seed'] = seed
    exp_name = 'w_AGN' if config['use_agn'] else 'w_bert'
    logger = setup_logging(exp_name)
    logger.info("Starting the training process.")
    logger.info("Experiment configuration: %s", json.dumps(config))
    pprint(config)

    # Create save directory if it doesn't exist
    if not os.path.exists(config['save_dir']):
        os.makedirs(config['save_dir'])

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_dir'], do_lower_case=True)

    # Load data and set datasets
    dataloader = CustomDataLoader(tokenizer,
                                  max_len=config['max_len'],
                                  ae_latent_dim=config['ae_latent_dim'],
                                  use_vae=True,
                                  batch_size=config["batch_size"],
                                  ae_epochs=config['ae_epochs'])

    dataloader.set_train(config['train_path'])
    dataloader.set_dev(config['dev_path'])
    dataloader.set_test(config['test_path'])

    dataloader.save_autoencoder(os.path.join(config['save_dir'], 'autoencoder.weights'))
    dataloader.save_vocab(os.path.join(config['save_dir'], 'vocab.pickle'))

    test_dataset = NERDataset(dataloader.test_set)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             collate_fn=collate_fn)

    micro_f1_list = []
    try:
        for idx in range(0, config['iterations']):
            print(f"Starting iteration {idx}...")

            train_dataset = NERDataset(dataloader.train_set)
            dev_dataset = NERDataset(dataloader.dev_set)


            # 创建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                      collate_fn=collate_fn)
            dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)


            # 更新配置参数
            config['steps_per_epoch'] = len(train_dataset) // config['batch_size']
            config['output_size'] = dataloader.label_size

            print('!!!>>>>> output size:', config['output_size'])

            # Model
            model = AGNModel(config, task='ner').to(device)

            metrics_callback = NERMetrics(
                model=model,
                eval_loader=dev_loader,
                save_path=os.path.join(config['save_dir'], 'best_model.weights'))

            print("Start training...")
            model.train()

            # 创建优化器
            optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
            # 创建调度器
            scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            # loss_function = torch.nn.CrossEntropyLoss()

            for epoch in tqdm(range(0, config['epochs'])):
                epoch_loss = 0  # 初始化损失累加变量
                count_batches = 0

                # 包裹 train_loader 使用 tqdm 创建进度条
                train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}', leave=True)
                for token_id, segment_ids, tfidf_vec, labels in train_pbar:
                    token_id = token_id.to(device)
                    segment_ids = segment_ids.to(device)
                    tfidf_vec = tfidf_vec.to(device)
                    labels = labels.to(device)

                    # 打印变量类型
                    # print(f"Type of token_id: {type(token_id)}")
                    # print(f"Type of segment_ids: {type(segment_ids)}")
                    # print(f"Type of tfidf_vec: {type(tfidf_vec)}")
                    # print(f"Type of labels: {type(labels)}")

                    optimizer.zero_grad()

                    emissions, attn_weights = model(token_id, segment_ids, tfidf_vec)  # 调用模型
                    loss = -model.crf(emissions, labels.long())  # 计算损失
                    epoch_loss += loss.item()
                    count_batches += 1

                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新权重

                    # 更新进度条的后缀显示当前批次的损失
                    train_pbar.set_postfix(loss="{:.4f}".format(loss.item()))

                average_loss = epoch_loss / count_batches  # 计算平均损失
                print(f"Epoch {epoch}/{config['epochs']}, Average Loss: {average_loss:.4f}")
                metrics_callback.on_epoch_end(epoch)
                print(
                    f"Epoch {epoch}/{config['epochs']}, Average Loss: {average_loss:.4f}, Micro F1: {max(metrics_callback.history['val_f1'])}")
                if metrics_callback.should_stop():
                    break
                log_experiment(logger, epoch, config, average_loss, metrics_callback)

            f1 = max(metrics_callback.history["val_f1"])

            micro_f1_list.append(f1)
            print(f"Iteration {idx}, F1: {f1}")

    except Exception as e:
        logger.error("An error occurred: %s", str(e))

    print("Average F1:", np.mean(micro_f1_list))
    logger.info("Training process completed.")
    # 在测试集上进行评估
    if config.get("is_test", False):
        model = AGNModel(config, task='ner').to(device)
        weights_path = os.path.join(config['save_dir'], 'best_model.weights')
        load_model_weights(model, weights_path)  # 加载模型权重

        test_metrics = NERTestMetrics(model, test_loader)
        test_f1,c_r = test_metrics.evaluate()
        logger.info(f"Completed evaluation on test set with Micro F1: {test_f1:.4f}")
        logger.info(f"Completed evaluation on test set with classification report: {c_r}")
    else:
        logger.info("Test evaluation skipped as per configuration.")
