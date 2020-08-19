"""
本文件使用pytorch，基于online shopping数据集，训练基本的双向双层lstm网络模型
"""

import os
import torch
import pandas as pd
import logging
from torch.utils.data import Dataset


logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class InputExample(object):
    """ 原始输入特征类。包括文本text与标签label """
    def __init__(self, guid, text, label=None):
        """Constructs a InputExample."""
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """ 原始输入转换后的特征类，包括文本切词后的词id，文本长度，标签 """
    def __init__(self, input_ids, segment_ids, label):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):
    """ 加载数据，生成元素为InputExample的列表 """
    def __init__(self, config, mode):
        self.mode = mode
        self.config = config

    def _create_examples(self, input_file, set_type):
        """ 生成InputExample列表 """
        examples = []
        data_df = pd.read_csv(input_file, sep='\t')
        for index, row in data_df.iterrows():
            guid = "%s-%s" % (set_type, index + 1)
            text_a = row['text_a']
            if self.mode != 'predict':
                label = row['polarity']
            else:
                label = None
            example = InputExample(guid=guid, text=text_a, label=label)
            examples.append(example)
        return examples

    def get_examples(self, single_predict_texts=None):
        if self.mode == 'train':
            train_examples = self._create_examples(os.path.join(self.config.data_path,
                                                                'online_shopping_train.tsv'), 'train')
            return train_examples
        elif self.mode == 'dev':
            dev_examples = self._create_examples(os.path.join(self.config.data_path,
                                                              'online_shopping_dev.tsv'), 'dev')
            return dev_examples
        elif self.mode == 'predict':
            predict_examples = self._create_examples(os.path.join(self.config.data_path,
                                                                  'online_shopping_predict.tsv'), 'predict')
            return predict_examples
        elif self.mode == 'single_predict':
            single_predict_examples = []
            for texts in single_predict_texts:
                example = InputExample(guid=single_predict_texts.index(texts), text=texts)
                single_predict_examples.append(example)
            return single_predict_examples
        else:
            raise Exception(' No Such Mode. ')

    @staticmethod
    def get_labels():
        return [0, 1]


def convert_single_example(ex_index, example, tokenizer):
    """ 将一个example类的训练数据转成feature类 """
    text = example.text

    tokens = tokenizer.tokenize(text)
    word_pieces = ['[CLS]'] + tokens

    len_text = len(word_pieces)

    input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
    segment_ids = [0] * len_text
    label = example.label

    # 打印前5条转换的记录
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("label: %s" % label)
    feature = InputFeatures(input_ids=input_ids, segment_ids=segment_ids, label=label)
    return feature


def convert_examples_to_features(examples, tokenizer):
    """ 将InputExample转为InputFeatures """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, tokenizer)
        features.append(feature)
    logging.info('Writing example Done.')
    return features


class OnlineShopping(Dataset):

    def __init__(self, mode, config, tokenizer, auto_padding=True, texts=None):
        self.processor = DataProcessor(config, mode)
        if mode != 'single_predict':
            examples = self.processor.get_examples()
        else:
            examples = self.processor.get_examples(texts)
        features = convert_examples_to_features(examples, tokenizer)

        self.max_len = config.max_len
        assert self.max_len <= 512
        if auto_padding:
            self.text_ids = [feature.input_ids[:512] for feature in features]
            self.segment_ids = [feature.segment_ids[:512] for feature in features]
        else:
            self.text_ids = [self.padding(feature.input_ids) for feature in features]
            self.segment_ids = [[0] * config.max_len for _ in features]
        self.labels = [self.convert_label_value_to_id(feature.label) if feature.label is not None else feature.label
                       for feature in features]

        if mode == 'train':
            self.num_steps = int(len(examples) / config.train_batch_size * config.num_epochs)
        elif mode == 'dev':
            self.num_steps = int(len(examples) / config.dev_batch_size)
        else:
            self.num_steps = None

    def padding(self, raw_text_id):
        while len(raw_text_id) < self.max_len:
            raw_text_id.append(0)
        if len(raw_text_id) > self.max_len:
            raw_text_id = raw_text_id[:self.max_len]
        return raw_text_id

    def convert_label_value_to_id(self, label):
        label_values = self.processor.get_labels()
        return label_values.index(label)

    def convert_label_id_to_value(self, label_id):
        label_values = self.processor.get_labels()
        return label_values[int(label_id)]

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, index):
        return self.text_ids[index], self.segment_ids[index], self.labels[index]


if __name__ == '__main__':

    import configuration
    import torch.nn.utils.rnn as rnn_utils
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer

    conf = configuration.Config()
    tokenizer = BertTokenizer.from_pretrained(conf.pretrained_model_name)
    dataset = OnlineShopping('train', conf, tokenizer, auto_padding=True)

    def collate_fn(batches):
        """ 每个batch做padding，而非所有样本做padding """
        input_ids = [torch.tensor(batch[0]) for batch in batches]
        input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)

        segment_ids = [torch.tensor(batch[1]) for batch in batches]
        segment_ids = rnn_utils.pad_sequence(segment_ids, batch_first=True, padding_value=0)

        masks_tensors = torch.zeros(input_ids.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(input_ids != 0, 1)

        if batches[0][2] is not None:
            labels = torch.tensor([batch[2] for batch in batches], dtype=torch.long)
        else:
            labels = None

        return input_ids, segment_ids, masks_tensors, labels

    train_dataloader = DataLoader(dataset,
                                  batch_size=5,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    data = next(iter(train_dataloader))
    ids, segments, masks, labels = data
    print(ids)
    print(ids.shape)
    print(segments)
    print(segments.shape)
    print(masks)
    print(masks.shape)
    print(labels)
    print(labels.shape)

