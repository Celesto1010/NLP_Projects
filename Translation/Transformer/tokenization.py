"""
本文件定义源文本字符的处理方法
"""

import tensorflow_datasets as tfds


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(en_vocab_file, zh_vocab_file):
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
    return subword_encoder_en, subword_encoder_zh


class FullTokenizer(object):
    """Runs end-to-end tokenization."""

    def __init__(self, en_vocab_file, zh_vocab_file):
        # 根据vocab文件，得到形如(词，index)的字典
        self.en_vocab, self.zh_vocab = load_vocab(en_vocab_file, zh_vocab_file)

    def get_en_vocab_size(self):
        return self.en_vocab.vocab_size

    def get_zh_vocab_size(self):
        return self.zh_vocab.vocab_size

    def convert_en_text_to_ids(self, en_text):
        return self.en_vocab.encode(en_text)

    def convert_zh_text_to_ids(self, zh_text):
        return self.zh_vocab.encode(zh_text)

    def convert_en_ids_to_text(self, en_ids):
        return self.en_vocab.decode(en_ids)

    def convert_zh_ids_to_text(self, zh_ids):
        return self.zh_vocab.decode(zh_ids)

    def show_en_ids_text_mapping(self, en_ids):
        for idx in en_ids:
            subword = self.en_vocab.decode([idx])
            print('{0:5}{1:6}'.format(idx, ' ' * 5 + subword))

    def show_zh_ids_text_mapping(self, zh_ids):
        for idx in zh_ids:
            subword = self.zh_vocab.decode([idx])
            print('{0:5}{1:6}'.format(idx, ' ' * 5 + subword))


if __name__ == '__main__':
    en_file = r'E:\NLP_Projects\Datasets\Translation_ZH-EN_wmt19\en_vocab'
    zh_file = r'E:\NLP_Projects\Datasets\Translation_ZH-EN_wmt19\zh_vocab'

    tokenizer = FullTokenizer(en_file, zh_file)
    print('en_vocab_size: ', tokenizer.get_en_vocab_size())
    print('zh_vocab_size: ', tokenizer.get_zh_vocab_size())

    # print(tokenizer.convert_zh_ids_to_text([6090]))

    english_text = "The eurozone’s collapse forces a major realignment of European politics."
    chinese_text = "欧元区的瓦解强迫欧洲政治进行一次重大改组。"

    english_ids = tokenizer.convert_en_text_to_ids(english_text)
    chinese_ids = tokenizer.convert_zh_text_to_ids(chinese_text)

    e_text = tokenizer.convert_en_ids_to_text(english_ids)
    z_text = tokenizer.convert_zh_ids_to_text(chinese_ids)

    print('原始中英原文：')
    print(english_text)
    print(chinese_text)
    print()
    print('转换后的中英id')
    print(english_ids)
    print(chinese_ids)
    print()
    print('id变回原文')
    print(e_text)
    print(z_text)