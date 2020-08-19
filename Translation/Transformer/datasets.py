import os
import torch
import json
import logging
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
    def __init__(self, config, mode):
        self.config = config
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
            train_examples = self._create_examples(os.path.join(self.config.file_config.data_path,
                                                                self.config.file_config.train_file), 'train')
            return train_examples
        elif self.mode == 'eval':
            eval_examples = self._create_examples(os.path.join(self.config.file_config.data_path,
                                                               self.config.file_config.eval_file), 'eval')
            return eval_examples
        elif self.mode == 'translate':
            predict_examples = self._create_examples(os.path.join(self.config.file_config.data_path,
                                                                  self.config.file_config.predict_file), 'translate')
            return predict_examples
        elif self.mode == 'demo':
            demo_examples = self._create_examples(os.path.join(self.config.file_config.data_path,
                                                               self.config.file_config.demo_file), 'demo')
            return demo_examples
        elif self.mode == 'single_translate':
            example = InputExample(guid='Translate', en_text=single_translate_texts, zh_text=None)
            return [example]
        else:
            raise Exception(' No Such Mode. ')


def convert_single_example(ex_index, example, tokenizer):
    """ 将一个example类的训练数据转成feature类 """
    en_text = example.en_text
    en_ids = tokenizer.convert_en_text_to_ids(en_text)
    # 给en_ids加上BOS与EOS
    en_ids = [tokenizer.en_vocab.vocab_size] + en_ids + [tokenizer.en_vocab.vocab_size + 1]

    zh_text = example.zh_text
    if zh_text is not None:
        zh_ids = tokenizer.convert_zh_text_to_ids(zh_text)
        # 给zh_ids加上BOS与EOS
        zh_ids = [tokenizer.zh_vocab.vocab_size] + zh_ids + [tokenizer.zh_vocab.vocab_size + 1]
    else:
        zh_ids = None

    # 打印前5条转换的记录
    if example.guid != 'Translate':
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s" % example.guid)
            logging.info("en_input_ids: %s" % " ".join([str(x) for x in en_ids]))
            if zh_ids is not None:
                logging.info("zh_input_ids: %s" % " ".join([str(x) for x in zh_ids]))
    feature = InputFeature(en_ids=en_ids, zh_ids=zh_ids)
    return feature


def convert_examples_to_features(examples, tokenizer, do_filter=False):
    """ 将InputExample转为InputFeatures """
    features = []
    for (ex_index, example) in enumerate(examples):
        if (ex_index + 1) % 10000 == 0:
            logging.info("Writing example %d of %d" % (ex_index + 1, len(examples)))
        feature = convert_single_example(ex_index, example, tokenizer)
        if do_filter:
            if len(feature.zh_ids) <= 40 and len(feature.en_ids) <= 40:
                features.append(feature)
        else:
            features.append(feature)
    if examples[0].guid != 'Translate':
        logging.info('Writing example Done. Total examples written: {}'.format(len(features)))
    return features


class Translate(Dataset):

    def __init__(self, mode, config, tokenizer, auto_padding=True, do_filter=False, texts=None):
        self.processor = DataProcessor(config, mode)
        if mode != 'single_translate':
            examples = self.processor.get_examples()
        else:
            examples = self.processor.get_examples(texts)
        features = convert_examples_to_features(examples, tokenizer, do_filter)

        self.max_len = config.train_config.max_len
        if auto_padding:
            self.en_ids = [feature.en_ids for feature in features]
            self.zh_ids = [feature.zh_ids for feature in features]
        else:
            self.en_ids = [self.padding(feature.en_ids) for feature in features]
            self.zh_ids = [self.padding(feature.zh_ids) for feature in features]

        if mode == 'train' or mode == 'demo':
            self.num_steps = int(len(examples) / config.train_config.train_batch_size * config.train_config.num_epochs)
        elif mode == 'eval':
            self.num_steps = int(len(examples) / config.train_config.eval_batch_size)
        else:
            self.num_steps = None

    def padding(self, raw_text_id):
        while len(raw_text_id) < self.max_len:
            raw_text_id.append(0)
        if len(raw_text_id) > self.max_len:
            raw_text_id = raw_text_id[:self.max_len]
        return raw_text_id

    def __len__(self):
        return len(self.en_ids)

    def __getitem__(self, index):
        return self.en_ids[index], self.zh_ids[index]


if __name__ == '__main__':
    import tokenization
    import configuration
    import torch.nn.utils.rnn as rnn_utils
    from torch.utils.data import DataLoader
    import models

    conf = configuration.Config()
    tokenizer = tokenization.FullTokenizer(en_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.en_vocab),
                                           zh_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.zh_vocab))
    conf.model_config.src_vocab_size = tokenizer.get_en_vocab_size() + 2
    conf.model_config.trg_vocab_size = tokenizer.get_zh_vocab_size() + 2
    dataset = Translate(mode='demo', config=conf, tokenizer=tokenizer, do_filter=False)
    transformer = models.Transformer(conf)

    def collate_fn(batches):
        """ 每个batch做padding，而非所有样本做padding """
        batch_en_ids = [torch.tensor(batch[0]) for batch in batches]
        batch_en_ids = rnn_utils.pad_sequence(batch_en_ids, batch_first=True, padding_value=0)

        if batches[0][1] is not None:
            batch_zh_ids = [torch.tensor(batch[1]) for batch in batches]
            batch_zh_ids = rnn_utils.pad_sequence(batch_zh_ids, batch_first=True, padding_value=0)
        else:
            batch_zh_ids = None
        return batch_en_ids, batch_zh_ids

    train_dataloader = DataLoader(dataset,
                                  batch_size=conf.train_config.train_batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    data = next(iter(train_dataloader))
    en_ids, zh_ids = data
    print(en_ids)
    print(en_ids.shape)
    print(zh_ids)
    print(zh_ids.shape)
    print()

    logits = transformer(en_ids, zh_ids[:, :-1])
    print(logits)
    print(logits.shape)             # [batch_size, zh_seq_len - 1, zh_vocab_size]

    # import torch.nn.functional as F
    # probs = F.softmax(logits, dim=2)
    # print(probs)
    # print(probs.shape)
    # print(torch.sum(probs[0, 0, :]))
    # print('------------')
    # translated_ids = torch.argmax(probs, dim=2)
    # print(translated_ids)
    # print(translated_ids.shape)
    # print('++++++++++')
    #
    # true_ids = zh_ids[:, 1:]
    # print(true_ids.shape)
    # print(true_ids)
    #

    print('====================')
    true_ids = zh_ids[:, 1:]
    logits = logits.reshape(-1, conf.model_config.trg_vocab_size)
    true_ids = true_ids.reshape(-1)
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    cost = loss(logits, true_ids)

    loss_masks = torch.ones(true_ids.shape, dtype=torch.long)
    loss_masks = loss_masks.masked_fill(true_ids == 0, 0)
    print(cost)
    print(cost.shape)
    print(loss_masks)
    print(loss_masks.shape)
    cost *= loss_masks
    print(cost)
    print(torch.mean(cost))
