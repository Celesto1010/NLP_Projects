import os


class ModelConfig(object):
    def __init__(self,
                 vocab_size=68355,
                 num_heads=4,
                 dim=256,
                 ff_size=1024,
                 drop_rate=0.2,
                 encoder_layers=3,
                 num_classes=2):
        self.num_heads = num_heads
        self.dim = dim
        self.ff_size = ff_size
        self.drop_rate = drop_rate
        self.encoder_layers = encoder_layers
        self.vocab_size = vocab_size
        self.num_classes = num_classes


class TrainConfig(object):
    """ 训练参数 """
    def __init__(self,
                 train_batch_size=10,
                 eval_batch_size=5,
                 predict_batch_size=5,
                 auto_padding=True,
                 num_epochs=5,
                 warmup_var=0.2,
                 max_len=128,
                 base_lr=1e-5,
                 max_lr=1e-3,
                 max_grad_norm=3,
                 model_dir=r'E:\NLP_Projects\Models\Classification_single_sequence\Transformer',
                 model_name='./transformer.pth'):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.auto_padding = auto_padding
        self.num_epochs = num_epochs
        self.warmup_var = warmup_var
        self.max_len = max_len
        self.base_lr = base_lr
        self.max_lr = max_lr
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


class Config(object):
    def __init__(self):
        self.model_config = ModelConfig()
        self.file_config = FileConfig()
        self.train_config = TrainConfig()
