# ====================================================
# CFG
# ====================================================
class CFG:
    wandb = True
    competition = 'lecr'
    debug = False
    apex = False
    print_freq = 200
    num_workers = 4
    #model = "microsoft/deberta-v3-base"
    model = "microsoft/mdeberta-v3-base"
    #model = "bert-base-multilingual-cased"
    #model = "xlm-roberta-base"
    #model = "paraphrase-multilingual-mpnet-base-v2-finetuned-1-28"
    #model = "output_simcse_model_epo60"
    #model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    #model = "all-MiniLM-L6-v2-finetuned"
    #model = "bert-large-multilingual-cased"

    gradient_checkpointing = True
    scheduler = 'cosine'  # ['linear', 'cosine']
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 0
    epochs = 10
    encoder_lr = 2e-5
    decoder_lr = 2e-5
    min_lr = 1e-6
    eps = 1e-6
    layerwise_learning_rate_decay = 0.9
    adam_epsilon = 1e-6

    betas = (0.9, 0.999)
    batch_size = 32
    max_len = 256
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    n_fold = 1
    trn_fold = [0]
    train = True


if CFG.debug:
    CFG.epochs = 2
    CFG.trn_fold = [0]

# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import shutil
import string
import pickle
import random
import joblib
import itertools
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_squared_error
import torch

print(f"torch.__version__: {torch.__version__}")
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

# os.system('pip uninstall -y transformers')
# os.system('pip uninstall -y tokenizers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels transformers')
# os.system('python -m pip install --no-index --find-links=../input/pppm-pip-wheels tokenizers')
import tokenizers
import transformers

print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from adv_utils import FGM, PGD, AWP, EMA
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from adv_utils import *

device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
# tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
# text = 'text topic' + '[SEP]' + 'text topic'
# text_2 = 'text topic' + '</s>' + 'text topic'
# inputs = tokenizer.encode_plus(text)
# inputs_2 = tokenizer.encode_plus(text_2)


INPUT_DIR = './data/'
OUTPUT_DIR = './output_model_class_first/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ====================================================
# Utils
# ====================================================
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
def get_score(y_trues, y_preds):
    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    f1 = f1_score(y_true=y_trues, y_pred=y_preds, average='macro')
    #f1 = f1_score(y_true=y_trues, y_pred=y_preds, average='binary')
    acc, f1 = round(acc, 4), round(f1, 4)
    return acc, f1

def f2_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    tp = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    fp = np.array([len(x[1] - x[0]) for x in zip(y_true, y_pred)])
    fn = np.array([len(x[0] - x[1]) for x in zip(y_true, y_pred)])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
    return round(f2.mean(), 4)

def get_score_f2(dev_path, props):
    print('reading ', dev_path)
    x_val = pd.read_csv(dev_path)
    correlations = pd.read_csv('./learning-equality-curriculum-recommendations/correlations.csv')
    x_val['score'] = props
    x_val = x_val.sort_values(['topic_id', 'score'], ascending=[True, False]).reset_index(drop=True)

    best_f2_score = 0
    best_thre = 0
    best_n_rec = 0

    for thres in [0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3]:
        for n_rec in [9, 10, 11, 13, 14, 15, 16, 18, 20]:
            x_val['predictions'] = np.where(x_val['score'] > thres, 1, 0)
            x_val1 = x_val[x_val['predictions'] == 1]
            x_val1 = x_val1.groupby(['topic_id'])['content_id'].unique().reset_index()
            x_val1['content_id'] = x_val1['content_id'].apply(lambda x: ' '.join(x[:n_rec]))
            x_val1.columns = ['topic_id', 'predictions']
            x_val0 = pd.Series(x_val['topic_id'].unique())
            x_val0 = x_val0[~x_val0.isin(x_val1['topic_id'])]
            x_val0 = pd.DataFrame({'topic_id': x_val0.values, 'predictions': ""})
            x_val_r = pd.concat([x_val1, x_val0], axis=0, ignore_index=True)
            x_val_r = x_val_r.merge(correlations, how='left', on='topic_id')
            score = f2_score(x_val_r['content_ids'], x_val_r['predictions'])

            print('Threshold:', thres, 'N_RECALLS', n_rec, 'Average F2:', score)

            if score >= best_f2_score:
                best_f2_score = score
                best_thre = thres
                best_n_rec = n_rec

    print('Eval over ... best_f2_score %.4f , best_thre %.4f, best_n_rec %.4f'%(best_f2_score, best_thre, best_n_rec))
    return best_f2_score, best_thre, best_n_rec


def get_score_f2_bak(dev_path, props):
    print('reading ', dev_path)
    dev_df_tmp = pd.read_csv(dev_path)
    df_target = pd.read_csv('./learning-equality-curriculum-recommendations/correlations.csv')
    dev_df_tmp['score'] = props
    dev_df_tmp = dev_df_tmp.sort_values(['topic_id', 'score'], ascending=[True, False]).reset_index(drop=True)

    best_f2_score = 0
    best_thre = 0
    best_n_rec = 0

    for thres in [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2]:
        for n_rec in [5, 7, 9, 12, 13, 15, 16, 18]:
            test_sub = dev_df_tmp[dev_df_tmp['score'] >= thres].reset_index(drop=True)
            sub_df = test_sub.groupby('topic_id').apply(lambda g: g.head(n_rec)).reset_index(drop=True)
            sub_df = sub_df[['topic_id', 'content_id']].groupby('topic_id')['content_id'].agg(list).to_frame(name='preds').reset_index()
            sub_df['preds'] = sub_df['preds'].apply(lambda x: ' '.join(x))

            df_test_metric = pd.merge(sub_df, df_target, on='topic_id', how='left')
            df_metric = df_test_metric[['content_ids', 'preds']].copy()
            df_metric['content_ids'] = df_metric['content_ids'].astype(str).apply(lambda x: x.split())
            df_metric['preds'] = df_metric['preds'].astype(str).apply(lambda x: x.split())
            f2_scores = []
            for _, row in df_metric.iterrows():
                true_content_ids = set(row['content_ids'])
                pred_content_ids = set(row['preds'])
                tp = len(true_content_ids.intersection(pred_content_ids))
                fp = len(pred_content_ids - true_content_ids)
                fn = len(true_content_ids - pred_content_ids)
                if pred_content_ids:
                    precision = tp / (tp + fp)
                    recall = tp / (tp + fn)
                    f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
                else:
                    f2 = 0
                f2_scores.append(f2)
            print('Threshold:', thres, 'N_RECALLS', n_rec, 'Average F2:', np.mean(f2_scores))

            if np.mean(f2_scores) >= best_f2_score:
                best_f2_score = np.mean(f2_scores)
                best_thre = thres
                best_n_rec = n_rec

    print('Eval over ... best_f2_score %.4f , best_thre %.4f, best_n_rec %.4f'%(best_f2_score, best_thre, best_n_rec))
    return best_f2_score, best_thre, best_n_rec

def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=2023)

def display(tmp):
    print(tmp)

# ====================================================
# Data Loading
# ====================================================
train_df = pd.read_csv('./train_df_class_simcse_re_100_first.csv')
DEV_PATH_DF = './dev_df_class_simcse_re_100_first.csv'
dev_df = pd.read_csv(DEV_PATH_DF)
# train_df = train_df.dropna()
# dev_df = dev_df.dropna()

print(f"train.shape: {train_df.shape}")
display(train_df.head())
print(f"dev.shape: {dev_df.shape}")
display(dev_df.head())

# ====================================================
# tokenizer
# ====================================================
tokenizer = AutoTokenizer.from_pretrained(CFG.model)
#tokenizer = AutoTokenizer.from_pretrained(CFG.model + '/tokenizer')
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

# ====================================================
# Define max_len
# ====================================================
# lengths = []
# for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
#     length = len(tokenizer(row['topic_text'], add_special_tokens=False)['input_ids'])
#     lengths.append(length)
#     length = len(tokenizer(row['content_text'], add_special_tokens=False)['input_ids'])
#     lengths.append(length)
#
# pd_tmp = pd.DataFrame()
# pd_tmp['Text_len'] = lengths
# print(pd_tmp['Text_len'].describe([.90, .95, .99, .995]))
# LOGGER.info(f"max_len: {CFG.max_len}")


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text_topic, text_content):
    text_topic = text_topic.replace('[SEP]', ' ')
    text_content = text_content.replace('[SEP]', ' ')
    text = text_topic + '[SEP]' + text_content
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=CFG.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text_topic = df['topic_text'].values
        self.text_content = df['content_text'].values
        self.labels = df['label'].values
        self.scores = df['score'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, f"{self.scores[item]:.4f} {self.text_topic[item]}", self.text_content[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


# ====================================================
# Model
# ====================================================
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            # self.config.hidden_dropout = 0.
            # self.config.hidden_dropout_prob = 0.
            # self.config.attention_dropout = 0.
            # self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # if self.cfg.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable

        self.pool = MeanPooling()
        self.fc_dropout = nn.Dropout(0.05)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs_topic = self.model(**inputs)
        last_hidden_states_topic = outputs_topic[0]
        feature = self.pool(last_hidden_states_topic, inputs['attention_mask'])
        logits = self.fc(self.fc_dropout(feature))

        return logits

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
             valid_loader, valid_labels, best_score, fgm, awp, ema_inst):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    save_step = int(len(train_loader) / 2)

    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            logits = model(inputs)
            loss = criterion(logits.view(-1), labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        ''' 
        # ---------------------fgm-------------
        fgm.attack(epsilon=1.0)  # embedding被修改了
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            logits = model(inputs)
            loss_avd = criterion(logits.view(-1), labels)
        if CFG.gradient_accumulation_steps > 1:
            loss_avd = loss_avd / CFG.gradient_accumulation_steps
        losses.update(loss_avd.item(), batch_size)
        scaler.scale(loss_avd).backward()
        fgm.restore()  # 恢复Embedding的参数
        # ---------------------fgm-------------
        '''

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()

            if ema_inst:
                ema_inst.update()

            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
        if CFG.wandb and step % 20 == 0:
            print({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})

        if (step + 1) % save_step == 0 and epoch > -1:
            if ema_inst:
                ema_inst.apply_shadow()

            # eval
            avg_val_loss, predictions, true_labels, props = valid_fn(valid_loader, model, criterion, device)
            # scoring
            acc, f1 = get_score(true_labels, predictions)
            print("EVAL acc %.4f, f1 %.4f"%(acc, f1))
            # best_f2_score, best_thre, best_n_rec = get_score_f2('./dev_df_random_50.csv', props)
            best_f2_score, best_thre, best_n_rec = get_score_f2(DEV_PATH_DF, props)

            LOGGER.info(f'Epoch {epoch + 1} - step: {step:.4f}  avg_val_loss: {avg_val_loss:.4f}')
            if CFG.wandb:
                print({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] f2_score": best_f2_score,
                       f"[fold{fold}] thre": best_thre,
                       f"[fold{fold}] n_rec": best_n_rec,
                       f"[fold{fold}] best_score": best_score})

            if best_f2_score >= best_score:
                best_score = best_f2_score
                LOGGER.info(f'Epoch {epoch + 1} - Save Best loss: {best_score:.4f} Model')
                torch.save({'model': model.state_dict()},
                            #'predictions': predictions},
                           OUTPUT_DIR + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth")
            if ema_inst:
                ema_inst.restore()

    return losses.avg, best_score


def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    props = []
    true_labels = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        true_labels.extend(labels)
        # print(true_labels)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits.view(-1), labels)
            props.append(logits.sigmoid().squeeze().to('cpu').numpy().reshape(-1))

        # print(props)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)

        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    prop_all = np.concatenate(props, axis=0)
    prop_all = prop_all.tolist()

    for p in prop_all:
        if p > 0.5:
            preds.append(1)
        else:
            preds.append(0)

    return losses.avg, preds, true_labels, prop_all

def get_optimizer_grouped_parameters(
    model, model_type,
    learning_rate, weight_decay,
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters

# ====================================================
# train loop
# ====================================================
def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    # train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    # valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    # valid_labels = valid_folds[CFG.target_cols].values

    train_folds = train_df
    valid_folds = dev_df
    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True)

    # model = CustomModel(cfg=None, config_path=CFG.model + '/config.pth', pretrained=False)
    # state = torch.load(CFG.model + '/mpnet_basev2_first_pretrain_fold0_best.pth',
    #                    map_location=torch.device('cpu'))
    # model.load_state_dict(state['model'])

    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        cfg.num_warmup_steps = 0
        #cfg.num_warmup_steps = num_train_steps * 0.0
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    # criterion = nn.SmoothL1Loss(reduction='mean')
    # #criterion = RMSELoss(reduction="mean")
    # criterion = CosineSimilarityLoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    #loss = loss_fn(rep_a=rep_a, rep_b=rep_b, label=label)

    best_score = 0
    fgm = FGM(model)
    awp = None
    ema_inst = EMA(model, 0.999)
    ema_inst.register()

    # ema_inst = None

    for epoch in range(CFG.epochs):
        start_time = time.time()
        # train
        valid_labels = None
        avg_loss, best_score = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
                                        valid_loader, valid_labels, best_score, fgm, awp, ema_inst)


    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


if __name__ == '__main__':

    def get_result(oof_df):
        labels = oof_df[CFG.target_cols].values
        preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')


    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train_df, fold)
