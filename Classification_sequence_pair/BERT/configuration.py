class Config(object):
    def __init__(self,
                 train_batch_size=10,
                 dev_batch_size=5,
                 num_epoch=3,
                 num_labels=3,
                 pretrained_model_name='bert-base-chinese',
                 model_dir=r'E:\NLP\Torch_Practices\models\Torch_FakeNews',
                 model_name='fake_news.pth'):
        self.train_batch_size = train_batch_size
        self.dev_batch_size = dev_batch_size
        self.num_epoch = num_epoch
        self.num_labels = num_labels
        self.pretrained_model_name = pretrained_model_name
        self.model_dir = model_dir
        self.model_name = model_name