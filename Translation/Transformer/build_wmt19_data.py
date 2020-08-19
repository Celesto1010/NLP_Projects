import json
from random import shuffle
import tensorflow_datasets as tfds
import os
import tensorflow as tf
tf.enable_eager_execution()


class WmtFromTfRecord(object):
    def __init__(self):
        self.data_raw_path = r'E:\NLP_Projects\Datasets\Translation_ZH-EN_wmt19'
        self.data_raw_file = 'raw_data.json'
        self.en_vocab = 'en_vocab'
        self.zh_vocab = 'zh_vocab'

        self.builder = self.get_builder()

    @staticmethod
    def get_builder():
        config = tfds.translate.wmt.WmtConfig(
            version="1.0.0",
            language_pair=("zh", "en"),
            subsets={
                tfds.Split.TRAIN: ["newscommentary_v14"],
            },
        )
        builder = tfds.builder("wmt_translate", config=config)
        return builder

    def build_example(self):
        examples = self.builder.as_dataset(split='train[0%:]', as_supervised=True)
        with open(os.path.join(self.data_raw_path, self.data_raw_file), 'w', encoding='utf-8') as f:
            for en, zh in examples:
                english_text = en.numpy().decode('utf-8')
                chinese_text = zh.numpy().decode('utf-8')
                dic = {'english': english_text, 'chinese': chinese_text}
                f.write(json.dumps(dic, ensure_ascii=False))
                f.write('\n')

    def create_split_files(self):
        raw_data = os.path.join(self.data_raw_path, self.data_raw_file)
        with open(raw_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            shuffle(lines)

            training_large = lines[: 200000]
            training = lines[:100000]
            validating = lines[200000: 210000]

            demo = lines[210000: 211000]

            with open(os.path.join(self.data_raw_path, 'train_large.json'), 'w', encoding='utf-8') as f0:
                for l in training_large:
                    f0.write(l)

            with open(os.path.join(self.data_raw_path, 'train.json'), 'w', encoding='utf-8') as f1:
                for t in training:
                    f1.write(t)

            with open(os.path.join(self.data_raw_path, 'dev.json'), 'w', encoding='utf-8') as f2:
                for v in validating:
                    f2.write(v)

            with open(os.path.join(self.data_raw_path, 'demo.json'), 'w', encoding='utf-8') as f3:
                for d in demo:
                    f3.write(d)

    def build_dictionary(self):
        """ 建立中英文词典，使用tfds中的SubwordTextEncoder """
        en_vocab_file = os.path.join(self.data_raw_path, self.en_vocab)
        zh_vocab_file = os.path.join(self.data_raw_path, self.zh_vocab)
        en_examples, zh_examples = [], []

        with open(os.path.join(self.data_raw_path, self.data_raw_file), 'r', encoding='utf-8') as t:
            lines = t.readlines()
            for line in lines:
                line = json.loads(line)
                en_examples.append(line['english'])
                zh_examples.append(line['chinese'])

        subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus((en for en in en_examples),
                                                                                     target_vocab_size=2 ** 13)
        subword_encoder_en.save_to_file(en_vocab_file)
        print(f"字典大小：{subword_encoder_en.vocab_size}")
        print(f"前 10 个 subwords：{subword_encoder_en.subwords[:10]}")
        print()

        subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus((zh for zh in zh_examples),
                                                                                     target_vocab_size=2 ** 13,
                                                                                     max_subword_length=1)
        subword_encoder_zh.save_to_file(zh_vocab_file)
        print(f"字典大小：{subword_encoder_zh.vocab_size}")
        print(f"前 10 個 subwords：{subword_encoder_zh.subwords[:10]}")
        print()
        return subword_encoder_en, subword_encoder_zh


if __name__ == '__main__':
    wmt = WmtFromTfRecord()
    wmt.build_dictionary()