import pandas as pd
import os
from train import DATA_DIR


# 1. 加载数据
data_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# 2. 清除空数据
empty_title = ((data_df['title2_zh'].isnull()) |
               (data_df['title1_zh'].isnull()) |
               (data_df['title2_zh'] == '') |
               (data_df['title2_zh'] == '0') |
               (data_df['label'].isnull()))
data_df = data_df[~empty_title]

# 3. 剔除过长样本
MAX_LENGTH = 30
data_df = data_df[~(data_df.title1_zh.apply(lambda x: len(x)) > MAX_LENGTH)]
data_df = data_df[~(data_df.title2_zh.apply(lambda x: len(x)) > MAX_LENGTH)]

# 只保留有效列并重命名
data_df = data_df.reset_index()
data_df = data_df.loc[:, ['title1_zh', 'title2_zh', 'label']]
data_df.columns = ['text_a', 'text_b', 'label']

# 以8：2切分训练集验证集
data_df.sample(frac=1).reset_index(drop=True)
total_len = len(data_df)
train_df = data_df[:int(total_len * 0.8)]
dev_df = data_df[int(total_len * 0.8):]
print('训练集共{0}条数据，验证集共{1}条数据'.format(len(train_df), len(dev_df)))

# 存储训练集与验证集
train_df.to_csv(os.path.join(DATA_DIR, 'train.tsv'), sep='\t', index=False)
dev_df.to_csv(os.path.join(DATA_DIR, 'dev.tsv'), sep='\t', index=False)