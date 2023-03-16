## 赛题链接 ：
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations

# 1. 数据说明
- valid_samples_1k_first.csv 文件为随机划分出来的 category != source 的 topic_id 留着做验证集

- learning-equality-curriculum-recommendations 目录下存放官方数据集

# 2. 操作步骤
- step1 代码为训练stage1 召回模型做数据准备, 将 train_flag 设为 True 和 False ，分别运行得到训练集和验证集

- step2 代码为训练 stage1 召回模型

- step3 代码为用召回模型对所有数据进行Top100 召回

- step4 代码为训练stage2 精排模型准备数据，将 train_flag 设为 True 和 False ，分别运行得到训练集和验证集

- step5 代码为训练stage2 精排二分类模型

## 详细 solution ：
https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394838
