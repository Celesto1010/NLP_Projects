import logging
import json
import os
from torch.utils.data import Dataset
import random
import torch
import transformers
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class InputExample(object):
    """ 原数据example类。为一篇文章 """
    def __init__(self, guid, text):
        self.guid = guid
        self.text = text


class InputFeature(object):
    """ 原数据feature类。为文章tokenize后的id """
    def __init__(self, text_ids):
        self.text_ids = text_ids


class PoetDataProcessor(object):
    """ 诗歌的数据预处理类，加载数据 """
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode

    @staticmethod
    def _create_example(input_file):
        examples = []
        with open(input_file, 'r', encoding='utf8') as f:
            logging.info(' ------- Reading Poets ------- ')
            # 这里lines的每一个元素均为一首诗。
            lines = json.load(f)
            # 若原文中有换行符，则用[SEP]替换
            for index, line in enumerate(lines):
                examples.append(InputExample(guid=index, text=line.replace('\n', ' [SEP] ')))
        return examples

    def get_examples(self):
        if self.mode == 'train':
            examples = self._create_example(os.path.join(self.data_dir, 'train_five.json'))
        elif self.mode == 'eval':
            examples = self._create_example(os.path.join(self.data_dir, 'eval_five.json'))
        else:
            raise Exception(' No Such Mode. ')
        return examples


class NovelDataProcessor(object):
    """ 小说的数据预处理类，加载数据 """


def convert_single_example(example, tokenizer):
    """ 对文章做tokenization并加上标志字符 """
    text = example.text
    text_tokens = tokenizer.tokenize(text)
    text_ids = tokenizer.convert_tokens_to_ids(text_tokens)

    text_ids.insert(0, tokenizer.convert_tokens_to_ids('[MASK]'))           # 文章开头添加MASK表示文章开始
    text_ids.append(tokenizer.convert_tokens_to_ids('[CLS]'))               # 文章之间添加CLS表示文章结束
    feature = InputFeature(text_ids=text_ids)
    return feature


def convert_examples_to_features(examples, tokenizer, min_length):
    """ 得到待喂入模型的features。限制最短长度 """
    features = []
    for example in tqdm(examples):
        # if ex_index % 10000 == 0:
        #     logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(example, tokenizer)
        if len(feature.text_ids) >= min_length:
            features.append(feature)
    logging.info('Writing example Done. Total examples written: {} .'.format(len(features)))
    return features


class GPT2Dataset(Dataset):
    def __init__(self, data_processor, mode, tokenizer, n_ctx=1024, stride=768, min_length=0):
        self.processor = data_processor
        self.n_ctx = n_ctx
        self.stride = stride
        assert self.n_ctx > self.stride, 'n_ctx should be larger than stride.'
        examples = self.processor.get_examples()
        features = convert_examples_to_features(examples, tokenizer, min_length)

        # text_ids的每一个元素都是原文章词tokenize后的id
        self.text_ids = [feature.text_ids for feature in features]
        # full_ids为一维列表
        self.full_ids = []
        for feature in features:
            self.full_ids.extend(feature.text_ids)
        self.samples = self.create_sample()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def create_sample(self):
        """ 创建一条sample数据
            1. 把所有数据拉成一条
            2. 以n_ctx值作为一个sample中的token数目
            3. 以stride值作为start point迭代的步长
            4. 打乱顺序
        """
        start_point = 0
        samples = []
        while start_point < len(self.full_ids) - self.n_ctx:
            # stride小于n_ctx，samples中的各个元素会包含重叠的token
            # n_ctx即一个sample中的token数目
            samples.append(self.full_ids[start_point: start_point + self.n_ctx])
            start_point += self.stride
        if start_point < len(self.full_ids):
            samples.append(self.full_ids[len(self.full_ids) - self.n_ctx:])
        return torch.tensor(samples)


if __name__ == '__main__':
    proc = PoetDataProcessor(data_dir='./data', mode='train')
    tokenizer = transformers.BertTokenizer.from_pretrained(
        r'E:\NLP_Projects\Models\Bert_Pretrained\Chinese\Bert_Tokenizer')
    data = GPT2Dataset(proc, tokenizer=tokenizer, mode='train')

    print(data[0])
    print(data.full_ids[:1024])
    print(tokenizer.convert_ids_to_tokens(data[0]))
    print('-' * 50)
    print(data[1])
    print(data.full_ids[768: 768 + 1024])
    print(tokenizer.convert_ids_to_tokens(data[1]))
