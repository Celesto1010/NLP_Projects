import os
import json
import logging
from tqdm import tqdm
from torch.utils.data import Dataset


logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class InputExample(object):
    def __init__(self, guid, en_text, zh_text):
        self.guid = guid
        self.en_text = en_text
        self.zh_text = zh_text


class InputFeature(object):
    def __init__(self, en_ids, zh_ids):
        self.en_ids = en_ids
        self.zh_ids = zh_ids


class DataProcessor(object):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode

    @staticmethod
    def _create_examples(input_file, set_type):
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                guid = "%s-%s" % (set_type, index + 1)
                line = json.loads(line)
                example = InputExample(guid=guid, en_text=line['english'], zh_text=line['chinese'])
                examples.append(example)
        return examples

    def get_examples(self, single_translate_texts=None):
        if self.mode == 'train':
            train_examples = self._create_examples(os.path.join(self.data_dir,
                                                                'train_large.json'), 'train')
            return train_examples
        elif self.mode == 'eval':
            eval_examples = self._create_examples(os.path.join(self.data_dir,
                                                               'eval.json'), 'eval')
            return eval_examples
        elif self.mode == 'translate':
            predict_examples = self._create_examples(os.path.join(self.data_dir,
                                                                  'translate.json'), 'translate')
            return predict_examples
        elif self.mode == 'demo':
            demo_examples = self._create_examples(os.path.join(self.data_dir,
                                                               'demo.json'), 'demo')
            return demo_examples
        elif self.mode == 'single_translate':
            example = InputExample(guid='Translate', en_text=single_translate_texts, zh_text=None)
            return [example]
        else:
            raise Exception(' No Such Mode. ')


def convert_single_example(ex_index, example, tokenizer):
    """ 将一个example类的训练数据转成feature类 """
    en_text = example.en_text
    en_ids = tokenizer.tokenizer_en.encode(en_text, add_special_tokens=True)      # 给开头加上[CLS]，结尾加上[SEP]
    en_ids = en_ids[:512]               # Bert限制的sequence length

    zh_text = example.zh_text
    if zh_text is not None:
        zh_ids = tokenizer.tokenizer_zh.encode(zh_text, add_special_tokens=True)
        zh_ids = zh_ids[:512]
    else:
        zh_ids = None

    # 打印前5条转换的记录
    if ex_index < 5:
        logging.info("*** Example ***")
        logging.info("guid: %s" % example.guid)
        logging.info("en_texts: %s" % en_text)
        logging.info("en_input_ids: %s" % " ".join([str(x) for x in en_ids]))
        if zh_ids is not None:
            logging.info("zh_texts: %s" % zh_text)
            logging.info("zh_input_ids: %s" % " ".join([str(x) for x in zh_ids]))
    feature = InputFeature(en_ids=en_ids, zh_ids=zh_ids)
    return feature


def convert_examples_to_features(examples, tokenizer, do_filter=False):
    """ 将InputExample转为InputFeatures """
    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        feature = convert_single_example(ex_index, example, tokenizer)
        if do_filter:
            if len(feature.zh_ids) <= 40 and len(feature.en_ids) <= 40:
                features.append(feature)
        else:
            features.append(feature)
    logging.info('Writing example Done. Total examples written: {}'.format(len(features)))
    return features


class Translate(Dataset):

    def __init__(self, mode, data_dir, tokenizer, do_filter=False, texts=None, max_len=128):
        self.processor = DataProcessor(data_dir, mode)
        if mode != 'single_translate':
            examples = self.processor.get_examples()
        else:
            examples = self.processor.get_examples(texts)
        features = convert_examples_to_features(examples, tokenizer, do_filter)

        self.en_ids = [feature.en_ids for feature in features]
        self.zh_ids = [feature.zh_ids for feature in features]

    def __len__(self):
        return len(self.en_ids)

    def __getitem__(self, index):
        return self.en_ids[index], self.zh_ids[index]

