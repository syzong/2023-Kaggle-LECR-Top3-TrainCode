MAX_SEQ_LEN = 256

import warnings
warnings.simplefilter('ignore')

import os
import gc
import math
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

train_flag = True


# 验证集 topic_id
valid_topic_id = pd.read_csv('data/valid_samples_1k_first.csv')
valid_topic_id = valid_topic_id['topic_id'].values.tolist()
df_topic = pd.read_csv('./learning-equality-curriculum-recommendations/topics.csv')
df_target = pd.read_csv(f'./learning-equality-curriculum-recommendations/correlations.csv')


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

if train_flag:
    df_topic_tmp = df_topic[df_topic['id'].isin(valid_topic_id)].copy().reset_index(drop=True)
    df_topic_tmp = df_topic_tmp[df_topic_tmp['category'] == 'source'].copy().reset_index(drop=True)

    df_topic = df_topic[~df_topic['id'].isin(valid_topic_id)].copy().reset_index(drop=True)

    df_topic = pd.concat([df_topic, df_topic_tmp])
else:
    df_topic = df_topic[df_topic['id'].isin(valid_topic_id)].copy().reset_index(drop=True)
    # 去掉线上没有的 source 类型
    df_topic = df_topic[df_topic['category'] != 'source'].copy().reset_index(drop=True)

print('df_topic.shape ', df_topic.shape)
print('df_target.shape ', df_target.shape)

train_topic_id = df_topic['id'].unique().tolist()

print(len(train_topic_id), len(valid_topic_id))
# 候选集

df_content = pd.read_csv('./learning-equality-curriculum-recommendations/content.csv')
print(len(df_content))

# 正样本, 从 correlations 获取

df_labels = pd.read_csv('./learning-equality-curriculum-recommendations/correlations.csv')

train_labels = df_labels[df_labels['topic_id'].isin(train_topic_id)].reset_index(drop=True)
train_labels.content_ids = train_labels.content_ids.str.split()
train_labels = train_labels.explode("content_ids").rename(columns={"content_ids": "content_id"})
train_labels['label'] = 1

df_topic_recall_knn = pd.read_csv('df_topic_recall_100_simcse_first.csv')
# 负样本, 从同一个 language 里随机挑选生成

lang_dict = df_content.groupby('language')['id'].agg(list).to_frame(name='content_id').to_dict()['content_id']
df = pd.DataFrame({'id': train_topic_id})
df = df.merge(df_topic[['id', 'language']], on='id', how='left')
df = df.merge(df_labels, left_on='id', right_on='topic_id', how='left')
df = df.merge(df_topic_recall_knn, on='id', how='left')

print('df.shape ', df.shape)

neg_samples = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    topic_id = row['id']
    lang = row['language']
    content_ids = row['content_ids']
    if pd.isna(row['recall_ids']):
        print(' Error 1 , no recall_ids ...')
        recall_ids_list = random.choices(lang_dict[lang], k=100)
    else:
        recall_ids_list = row['recall_ids'].split()

    if pd.isna(content_ids):
        print(' Error 2 , no label content_ids ...')
        for id in recall_ids_list:
            neg_samples.append({'topic_id': topic_id, 'content_id': id, 'label': 0})
    else:
        contents = content_ids.split()
        for id in recall_ids_list:
            if id not in contents:
                neg_samples.append({'topic_id': topic_id, 'content_id': id, 'label': 0})
            else:
                neg_samples.append({'topic_id': topic_id, 'content_id': id, 'label': 1})
        
neg_df = pd.DataFrame(neg_samples)

if train_flag:
    # 合并正负样本
    train_labels = pd.concat([train_labels, neg_df]).sort_values('topic_id').reset_index(drop=True)
else:
    train_labels = neg_df

print(train_labels)
print(train_labels.label.value_counts())

print(" drop_duplicates ..")
train_labels = train_labels.drop_duplicates()
print(train_labels)
print(train_labels.label.value_counts())

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
df_topic.fillna('', inplace=True)

# 拼接 children 和 parent description 文本
# https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/376873

def get_text_content(row):
    text = row['title'] +\
           '[SEP]' + row['description'] +\
           '[SEP]' + row['text']
    return text[:MAX_SEQ_LEN]


def get_text_topic(row):
    text = row['title'] +\
           '[SEP]' + row['description'] +\
           '[SEP]' + row['context'] +\
           '[SEP]' + row['parent_description'] +\
           '[SEP]' + row['children_description']
    return text[:MAX_SEQ_LEN]


df_content['content_text'] = df_content.apply(lambda row: get_text_content(row), axis=1)
df_topic['topic_text'] = df_topic.apply(lambda row: get_text_topic(row), axis=1)

#生成训练集
content_df = df_content[['id', 'content_text']].copy()
content_df.columns = ['content_id', 'content_text']
topic_df = df_topic[['id', 'topic_text']].copy()
topic_df.columns = ['topic_id', 'topic_text']
df_train = pd.merge(train_labels, topic_df, on='topic_id', how='left')
df_train = pd.merge(df_train, content_df, on='content_id', how='left')
print(df_train)

if train_flag:
    df_train.to_csv('train_df_class_simcse_re_100_first.csv', index=None)
else:
    df_train.to_csv('dev_df_class_simcse_re_100_first.csv', index=None)
