class ModelConfig(object):
    def __init__(self,
                 num_heads=8,
                 dim=512,
                 ff_size=2048,
                 drop_rate=0.2,
                 encoder_layers=6,
                 decoder_layers=6,
                 src_vocab_size=None,
                 trg_vocab_size=None):
        self.num_heads = num_heads
        self.dim = dim
        self.ff_size = ff_size
        self.drop_rate = drop_rate
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size


class FileConfig(object):
    def __init__(self):
        self.data_path = r'E:\NLP_Projects\Datasets\Translation_ZH-EN_wmt19'
        # self.data_path = '/tmp/wyd_tmp/translation/Translation_ZH-EN_zh2019'
        self.en_vocab = 'en_vocab'
        self.zh_vocab = 'zh_vocab'
        self.train_file = 'train.json'
        self.eval_file = 'eval.json'
        self.demo_file = 'demo.json'


class TrainConfig(object):
    def __init__(self,
                 train_batch_size=20,
                 eval_batch_size=10,
                 num_epochs=50,
                 max_len=128,
                 warmup_var=0.2,
                 base_lr=1e-6,
                 max_lr=1e-4,
                 auto_padding=True,
                 max_grad_norm=3.0,
                 model_dir=r'E:\NLP_Projects\Models\Translation\Transformer',
                 # model_dir='/tmp/wyd_tmp/translation/model',
                 model_name='transformer'):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_epochs = num_epochs
        self.max_len = max_len
        self.warmup_var = warmup_var
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.auto_padding = auto_padding
        self.max_grad_norm = max_grad_norm
        self.model_dir = model_dir
        self.model_name = model_name


class Config(object):
    def __init__(self):
        self.model_config = ModelConfig()
        self.file_config = FileConfig()
        self.train_config = TrainConfig()
