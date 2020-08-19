"""
本文件为配置文件
TrainConfig：训练参数
FileConfig：文件路径参数
ModelConfig：模型参数
"""

import os


class TrainConfig(object):
    """ 训练参数 """
    def __init__(self,
                 train_batch_size=20,
                 eval_batch_size=5,
                 predict_batch_size=5,
                 auto_padding=True,
                 num_epochs=4,
                 warmup_var=0.2,
                 max_len=128,
                 base_lr=1e-5,
                 max_lr=1e-3,
                 lr=0.1,
                 max_grad_norm=3,
                 model_dir=r'E:\NLP_Projects\Models\Classification_single_sequence\BiLSTM',
                 model_name='./bilstm.pth'):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.auto_padding = auto_padding
        self.num_epochs = num_epochs
        self.warmup_var = warmup_var
        self.max_len = max_len
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.model_dir = model_dir
        self.model_name = model_name


class FileConfig(object):
    """ 文件路径参数 """
    def __init__(self):
        self.data_path = r'E:\NLP_Projects\Datasets\Classification_ZH_OnlineShopping'
        self.train_file = os.path.join(self.data_path, './online_shopping_train.tsv')
        self.dev_file = os.path.join(self.data_path, './online_shopping_dev.tsv')
        self.stopwords_file = os.path.join(self.data_path, './stopwords.txt')
        self.vocab_file = os.path.join(self.data_path, './vocab.txt')


class ModelConfig(object):
    """ 模型参数 """
    def __init__(self,
                 vocab_size,                    # 词典中的词数
                 emb_size=128,                  # emb size
                 hidden_size=128,               # hidden size
                 drop_prob=0.1,
                 embedding_drop_prob=0.2,       # 词向量不被dropout的比例
                 num_of_classes=2,              # 分类数
                 num_of_layers=2,               # lstm网络层数
                 bidirectional=True,
                 initializer_range=0.02):       # 初始化范围
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.embedding_drop_prob = embedding_drop_prob
        self.num_of_classes = num_of_classes
        self.num_of_layers = num_of_layers
        self.bidirectional = bidirectional
        self.initializer_range = initializer_range


class LstmConfig(object):

    def __init__(self, vocab_size):
        self.train_config = TrainConfig()
        self.file_config = FileConfig()
        self.model_config = ModelConfig(vocab_size)