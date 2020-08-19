"""
本文件为配置文件
TrainConfig：训练参数
FileConfig：文件路径参数
ModelConfig：模型参数
"""


class Config(object):
    def __init__(self,
                 auto_padding=True,
                 train_batch_size=10,
                 dev_batch_size=5,
                 num_epochs=3,
                 num_labels=2,
                 max_len=128,
                 pretrained_model_name='bert-base-chinese',
                 pretrained_model_path=r'E:\NLP_Projects\Models\BERT_pretrained',
                 data_path=r'E:\NLP_Projects\Datasets\Classification_ZH_OnlineShopping',
                 model_dir=r'E:\NLP_Projects\Models\Classification_single_sequence\BERT',
                 model_name='bert.pth'):
        self.auto_padding = auto_padding
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.num_epochs = num_epochs
        self.num_labels = num_labels
        self.max_len = max_len
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_path = pretrained_model_path
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_name = model_name
