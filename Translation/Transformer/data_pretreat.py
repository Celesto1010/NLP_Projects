import json
import os
from random import shuffle
import tensorflow_datasets as tfds


path = r'E:\NLP_Projects\Datasets\Translation_ZH-EN_zh2019'
train = os.path.join(path, 'translation2019zh_train.json')

en_vocab = 'en_vocab'
zh_vocab = 'zh_vocab'


def create_split_files():
    """ 基于原始文档，取10w条作为训练、1w条作为测试、1k条作为demo"""

    with open(train, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        shuffle(lines)

        training_large = lines[300000: 1000000]
        training = lines[:100000]
        validating = lines[100000: 110000]

        demo = lines[200000: 201000]

        with open(os.path.join(path, 'train_large.json'), 'w', encoding='utf-8') as f0:
            for l in training_large:
                f0.write(l)

        with open(os.path.join(path, 'train.json'), 'w', encoding='utf-8') as f1:
            for t in training:
                f1.write(t)

        with open(os.path.join(path, 'dev.json'), 'w', encoding='utf-8') as f2:
            for v in validating:
                f2.write(v)

        with open(os.path.join(path, 'demo.json'), 'w', encoding='utf-8') as f3:
            for d in demo:
                f3.write(d)


def file_validate():
    """ 验证文档的读取 """
    with open(os.path.join(path, 'demo.json'), 'r', encoding='utf-8') as a:
        lines = a.readlines()
        for line in lines:
            line = json.loads(line)
            print(line['english'])
            print(line['chinese'])
            print()


def build_dictionary():
    """ 建立中英文词典，使用tfds中的SubwordTextEncoder """
    en_vocab_file = os.path.join(path, en_vocab)
    zh_vocab_file = os.path.join(path, zh_vocab)
    en_examples, zh_examples = [], []

    with open(os.path.join(path, 'train.json'), 'r', encoding='utf-8') as t:
        lines = t.readlines()
        for line in lines:
            line = json.loads(line)
            en_examples.append(line['english'])
            zh_examples.append(line['chinese'])

    subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en for en in en_examples),
                                                                                 target_vocab_size=2 ** 13)
    subword_encoder_en.save_to_file(en_vocab_file)
    print(f"字典大小：{subword_encoder_en.vocab_size}")
    print(f"前 10 個 subwords：{subword_encoder_en.subwords[:10]}")
    print()

    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus((zh for zh in zh_examples),
                                                                                 target_vocab_size=2 ** 13,
                                                                                 max_subword_length=1)
    subword_encoder_zh.save_to_file(zh_vocab_file)
    print(f"字典大小：{subword_encoder_zh.vocab_size}")
    print(f"前 10 個 subwords：{subword_encoder_zh.subwords[:10]}")
    print()
    return subword_encoder_en, subword_encoder_zh