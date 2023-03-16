TOP_K = 5
N_RECALLS = 100
MAX_SEQ_LEN = 256


MODEL_NAME = "output_simcse_model"

import warnings
warnings.simplefilter('ignore')

import os
import re
import gc
import sys
import multiprocessing

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding


# 加载数据

DATA_DIR = './learning-equality-curriculum-recommendations'

# 验证集 topic_id
valid_topic_id = pd.read_csv('data/valid_samples_1k_first.csv')
valid_topic_id = valid_topic_id['topic_id'].values.tolist()

df_content = pd.read_csv(f'{DATA_DIR}/content.csv')
df_topic = pd.read_csv(f'{DATA_DIR}/topics.csv')
df_target = pd.read_csv(f'{DATA_DIR}/correlations.csv')

df_tmp = df_topic[['id', 'parent']].reset_index(drop=True).copy()
df_tmp.columns = ['child', 'id']
df_topic = pd.merge(df_topic, df_tmp, on='id', how='left')
df_children = df_topic.groupby('id')['child'].agg(list).reset_index(name='children')
df_topic = df_topic.merge(df_children, on='id', how='left').drop_duplicates('id').reset_index(drop=True)
df_topic.drop('child', axis=1, inplace=True)
desc_dict = df_topic[['id', 'description']].fillna('').set_index('id').to_dict()['description']
parent_texts = []
children_texts = []
for _, row in tqdm(df_topic.iterrows(), total=len(df_topic)):
    desc = row['description']
    parent = row['parent']
    children = row['children']
    p_text = ''
    if not pd.isna(parent):
        p_text = desc_dict[parent]
    parent_texts.append(p_text)
    children_texts.append(' '.join([desc_dict[child] for child in children if not pd.isna(child)]))

df_topic['parent_description'] = parent_texts
df_topic['children_description'] = children_texts

df_topic = df_topic[df_topic['id'].isin(df_target['topic_id'].values.tolist())].reset_index(drop=True)

# 排除 4 个 language == 'mul' 不在 df_content 内的样本
df_topic = df_topic[df_topic['language'] != 'mul'].reset_index(drop=True)

# # 验证集
# df_topic = df_topic[df_topic['id'].isin(valid_topic_id)].copy().reset_index(drop=True)
# df_topic = df_topic[df_topic['category'] != 'source'].reset_index(drop=True)

df_content.shape, df_topic.shape, df_target.shape

# 文本预处理
# topic context

topics_df = pd.read_csv("./learning-equality-curriculum-recommendations/topics.csv",
                        index_col=0).fillna({"title": "", "description": ""})
content_df = pd.read_csv("./learning-equality-curriculum-recommendations/content.csv",
                         index_col=0).fillna("")

class Topic:
    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics_df.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    @property
    def content(self):
        if self.id in correlations_df.index:
            return [ContentItem(content_id) for content_id in correlations_df.loc[self.id].content_ids.split()]
        else:
            return tuple([]) if self.has_content else []

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        return separator.join(reversed([a.title for a in ancestors]))

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics_df[topics_df.parent == self.id].index]

    def subtree_markdown(self, depth=0):
        markdown = "  " * depth + "- " + self.title + "\n"
        for child in self.children:
            markdown += child.subtree_markdown(depth=depth + 1)
        for content in self.content:
            markdown += ("  " * (depth + 1) + "- " + "[" + content.kind.title() + "] " + content.title) + "\n"
        return markdown

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f"<Topic(id={self.id}, title=\"{self.title}\")>"


class ContentItem:
    def __init__(self, content_id):
        self.id = content_id

    @property
    def topics(self):
        return [Topic(topic_id) for topic_id in
                topics_df.loc[correlations_df[correlations_df.content_ids.str.contains(self.id)].index].index]

    def __getattr__(self, name):
        return content_df.loc[self.id][name]

    def __str__(self):
        return self.title

    def __repr__(self):
        return f"<ContentItem(id={self.id}, title=\"{self.title}\")>"

    def __eq__(self, other):
        if not isinstance(other, ContentItem):
            return False
        return self.id == other.id

    def get_all_breadcrumbs(self, separator=" >> ", include_root=True):
        breadcrumbs = []
        for topic in self.topics:
            new_breadcrumb = topic.get_breadcrumbs(separator=separator, include_root=include_root)
            if new_breadcrumb:
                new_breadcrumb = new_breadcrumb + separator + self.title
            else:
                new_breadcrumb = self.title
            breadcrumbs.append(new_breadcrumb)
        return breadcrumbs


def get_context(topic_id):
    topic = Topic(topic_id)
    return topic.get_breadcrumbs()


df_topic['context'] = df_topic['id'].apply(get_context)

del content_df, topics_df
gc.collect()
df_content.fillna('', inplace=True)

df_content.fillna('', inplace=True)
df_topic.fillna('', inplace=True)

def get_text_content(row):
    text = row['title'] + \
           ' [SEP] ' + row['kind'] + \
           ' [SEP] ' + row['language'] + \
           ' [SEP] ' + row['description'] + \
           ' [SEP] ' + row['text']
    return text[:MAX_SEQ_LEN]


def get_text_topic(row):
    ch_desc_text = ' '.join(row['children_description'])
    text = row['title'] + \
           ' [SEP] ' + row['channel'] + \
           ' [SEP] ' + row['category'] + \
           ' [SEP] ' + str(row['level']) + \
           ' [SEP] ' + str(row['language']) + \
           ' [SEP] ' + row['description'] + \
           ' [SEP] ' + row['context'] + \
           ' [SEP] ' + row['parent_description'] + \
           ' [SEP] ' + ch_desc_text
    return text[:MAX_SEQ_LEN]



df_content['text2'] = df_content.apply(lambda row: get_text_content(row), axis=1)
df_topic['text2'] = df_topic.apply(lambda row: get_text_topic(row), axis=1)


# language 划分
languages = df_content['language'].unique().tolist()

content_dict = {}
for lang in tqdm(languages):
    content_dict[lang] = df_content[df_content['language'] == lang].reset_index(drop=True)

# 加载预训练模型

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
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        # if self.cfg.gradient_checkpointing:
        #     self.model.gradient_checkpointing_enable

        self.pool = MeanPooling()
        self.fc_dropout = nn.Dropout(0.1)
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
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME + '/tokenizer')

model = CustomModel(cfg=None, config_path=MODEL_NAME + '/config.pth', pretrained=False)
state = torch.load(MODEL_NAME + '/sentence-transformers-paraphrase-multilingual-mpnet-base-v2_fold0_best.pth',
                   map_location=torch.device('cpu'))
model.load_state_dict(state['model'])

device = torch.device('cuda:1') if torch.cuda.device_count() > 1 else torch.device('cuda:0')
model.eval()
model.to(device)

###---------------------

class TestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        # text = self.texts[item].replace('[SEP]', '</s>')
        inputs = tokenizer(text,
                               add_special_tokens=True,
                               return_offsets_mapping=False)

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs

def get_model_feature(model, texts):
    feature_outs_all = []
    test_dataset = TestDataset(texts)
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             num_workers=0, pin_memory=True, drop_last=False)

    # tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in test_loader:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feature_outs = model(inputs)
            feature_outs_all.append(feature_outs)

    feature_outs_all_final = torch.cat(feature_outs_all, dim=0)
    print(feature_outs_all_final.shape)

    return feature_outs_all_final


corpus_embeddings_dict = {}
for lang in tqdm(languages):
    corpus_embeddings = get_model_feature(model, content_dict[lang]['text2'])
    corpus_embeddings_dict[lang] = corpus_embeddings

topic_embedding_list = get_model_feature(model, df_topic['text2'])

pred_final = []
for idx, row in tqdm(df_topic.iterrows(), total=len(df_topic)):
    query = row['text2']
    lang = row['language']
    if lang in corpus_embeddings_dict:
        corpus_embeddings = corpus_embeddings_dict[lang]
        content_df = content_dict[lang]
    else:
        corpus_embeddings = corpus_embeddings_dict['en']
        content_df = content_dict['en']

    query_embedding = topic_embedding_list[idx, :]

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min([N_RECALLS, len(corpus_embeddings)])
    top_results = torch.topk(cos_scores, k=top_k)

    indics = top_results[1].cpu().numpy()

    # threshold = 0.8
    # score_top = top_results[0].cpu().numpy()
    # in_use = np.where(score_top > threshold)
    # indics = indics[in_use]

    #pid = content_dict[lang]['id'][indics]
    pid = content_df['id'][indics]
    pred_final.append(' '.join(pid))

df_topic['recall_ids'] = pred_final


df_topic[['id', 'recall_ids']].to_csv('df_topic_recall_100_simcse_first.csv', index=None)


# 算分环节
df_metric = pd.merge(df_topic, df_target, left_on='id', right_on='topic_id', how='left')
# df_metric = df_metric[df_metric['content_ids'].notna()].reset_index(drop=True)
df_metric = df_metric[['topic_id', 'content_ids', 'recall_ids']].copy()

def get_pos_score(y_true, y_pred, top_n):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()[:top_n]))
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 50)
print(f'Our max positive score top 50 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 70)
print(f'Our max positive score top 70 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 100)
print(f'Our max positive score top 100 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 150)
print(f'Our max positive score top 150 is {pos_score}')

pos_score = get_pos_score(df_metric['content_ids'], df_metric['recall_ids'], 200)
print(f'Our max positive score top 200 is {pos_score}')

df_metric['content_ids'] = df_metric['content_ids'].astype(str).apply(lambda x: x.split())
df_metric['recall_ids'] = df_metric['recall_ids'].astype(str).apply(lambda x: x.split())
f2_scores = []

N_RECALLS = [50, 100, 200, 300, 400, 500, 600]
N_TOP_F2 = [5, 10, 15]
for n_top in N_TOP_F2:
    for _, row in tqdm(df_metric.iterrows(), total=len(df_metric)):
        true_ids = set(row['content_ids'])
        pred_ids = set(row['recall_ids'][:n_top])
        tp = len(true_ids.intersection(pred_ids))
        fp = len(pred_ids - true_ids)
        fn = len(true_ids - pred_ids)
        if pred_ids:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f2 = tp / (tp + 0.2 * fp + 0.8 * fn)
        else:
            f2 = 0
        f2_scores.append(f2)
    print(f'Average F2@{n_top}:', np.mean(f2_scores))
for n_recall in N_RECALLS:
    total = 0
    correct = 0
    for _, row in tqdm(df_metric.iterrows(), total=len(df_metric)):
        y_trues = row['content_ids']
        y_preds = row['recall_ids'][:n_recall]
        for y_true in y_trues:
            total += 1
            if y_true in y_preds:
                correct += 1
    print(f'hitrate@{n_recall}:', correct/total)
