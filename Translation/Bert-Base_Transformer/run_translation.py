import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

import time
import logging
import datasets
import os
import argparse
import transformers

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='variables for torch classification')

# 运行模式
parser.add_argument('--mode', type=str, default='train', help='run model training or run prediction')
parser.add_argument('--train', type=bool, default=True, help='if_training')
parser.add_argument('--eval', type=bool, default=True, help='if_validating')

# 训练参数
parser.add_argument('--num_epochs', type=int, default=10, help='num_epoch')
parser.add_argument('--train_batch_size', type=int, default=10, help='train_batch_size')
parser.add_argument('--warmup_steps', type=int, default=2000, help='warmup_steps')
parser.add_argument('--lr', type=float, default=1e-5, help='train_batch_size')

# 文件夹地址
parser.add_argument('--tokenizer_en_dir', type=str, required=True)
parser.add_argument('--tokenizer_zh_dir', type=str, required=True)
parser.add_argument('--model_config_dir', type=str, required=True)
parser.add_argument('--pretrained_english_dir', type=str, required=True)
parser.add_argument('--pretrained_chinese_dir', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokenizer(object):
    def __init__(self):
        self.tokenizer_en, self.tokenizer_zh = self.get_tokenizer()

    @staticmethod
    def get_tokenizer():
        tokenizer_en_dir = args.tokenizer_en_dir
        tokenizer_zh_dir = args.tokenizer_zh_dir

        if not os.path.exists(tokenizer_en_dir):
            logging.info(' *** Downloading Bert English Tokenizer *** ')
            os.mkdir(tokenizer_en_dir)
            os.mkdir(os.path.join(tokenizer_en_dir, 'cache'))
            tokenizer_en = transformers.BertTokenizer.from_pretrained('bert-base-uncased',
                                                                      cache_dir=os.path.join(tokenizer_en_dir, 'cache'))
            tokenizer_en.save_pretrained(tokenizer_en_dir)
        else:
            logging.info(' *** Loading Bert English Tokenizer *** ')
            tokenizer_en = transformers.BertTokenizer.from_pretrained(tokenizer_en_dir)

        if not os.path.exists(tokenizer_zh_dir):
            logging.info(' *** Downloading Bert Chinese Tokenizer *** ')
            os.mkdir(tokenizer_zh_dir)
            os.mkdir(os.path.join(tokenizer_zh_dir, 'cache'))
            tokenizer_zh = transformers.BertTokenizer.from_pretrained('bert-base-chinese',
                                                                      cache_dir=os.path.join(tokenizer_zh_dir, 'cache'))
            tokenizer_zh.save_pretrained(tokenizer_zh_dir)
        else:
            logging.info(' *** Loading Bert Chinese Tokenizer *** ')
            tokenizer_zh = transformers.BertTokenizer.from_pretrained(tokenizer_zh_dir)

        return tokenizer_en, tokenizer_zh


def get_model_config():
    """ 加载encoder-decoder模型参数 """
    ED_CONFIG = args.model_config_dir
    if not os.path.exists(ED_CONFIG):
        logging.info(' *** Downloading Encoder-Decoder Configuration *** ')
        os.mkdir(ED_CONFIG)
        config_encoder = transformers.BertConfig.from_pretrained('bert-base-uncased')
        config_decoder = transformers.BertConfig.from_pretrained('bert-base-chinese')

        config_ed = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
        config_ed.save_pretrained(ED_CONFIG)
    else:
        logging.info(' *** Loading Encoder-Decoder Configuration *** ')
        config_ed = transformers.EncoderDecoderConfig.from_pretrained(os.path.join(ED_CONFIG, 'config.json'))
    return config_ed


def get_pretrained_model(mode, model_config, load_epoch=None):
    """ 加载模型。
        mode=train: 基于原始的中英文两个预训练模型执行训练
        mode=other: 基于迁移学习后得到的模型进行继续训练、验证或预测"""
    # EncoderDecoderModel使用的是基本的BertModel. Encoder用英文的预训练模型，Decoder用中文的预训练模型
    if mode == 'train':
        logging.info(' *** Under training mode. Prepare to load pretrained bert model *** ')
        pretrained_english_dir = args.pretrained_english_dir
        pretrained_chinese_dir = args.pretrained_chinese_dir
        if not os.path.exists(pretrained_english_dir):
            logging.info(' *** Downloading English Pretrained BertModel *** ')
            os.mkdir(pretrained_english_dir)
            os.mkdir(os.path.join(pretrained_english_dir, 'cache'))
            bert_model = transformers.BertModel.from_pretrained('bert-base-uncased',
                                                                cache_dir=os.path.join(pretrained_english_dir,
                                                                                       './cache'))
            bert_model.save_pretrained(pretrained_english_dir)
        if not os.path.exists(pretrained_chinese_dir):
            logging.info(' *** Downloading Chinese Pretrained BertModel *** ')
            os.mkdir(pretrained_chinese_dir)
            os.mkdir(os.path.join(pretrained_chinese_dir, 'cache'))
            bert_model = transformers.BertForMaskedLM.from_pretrained('bert-base-chinese',
                                                                      cache_dir=os.path.join(pretrained_chinese_dir,
                                                                                             './cache'))
            bert_model.save_pretrained(pretrained_chinese_dir)

        ed_model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(pretrained_english_dir,
                                                                                    pretrained_chinese_dir,
                                                                                    config=model_config)
    else:
        logging.info(' *** Under other mode. Prepare to load trained model after epoch {} *** '.format(load_epoch))
        ed_model = transformers.EncoderDecoderModel.from_pretrained(
            os.path.join(args.model_dir, 'model_epoch_{}'.format(load_epoch)))
    return ed_model


def collate_fn(batches):
    """ 每个batch做padding，而非所有样本做padding
        每一组data，第0位是en_ids，第1位是zh_ids
        对于encoder-decoder模型，不需要segment_ids """
    batch_en_ids = [torch.tensor(batch[0]) for batch in batches]
    batch_en_ids = rnn_utils.pad_sequence(batch_en_ids, batch_first=True, padding_value=0)

    # 在bert里，pad部分mask标0（与自己实现的transformer相反）
    batch_en_masks_tensors = torch.zeros(batch_en_ids.shape, dtype=torch.long)
    batch_en_masks_tensors = batch_en_masks_tensors.masked_fill(batch_en_ids != 0, 1)

    if batches[0][1] is not None:
        batch_zh_ids = [torch.tensor(batch[1]) for batch in batches]
        batch_zh_ids = rnn_utils.pad_sequence(batch_zh_ids, batch_first=True, padding_value=0)

        batch_zh_masks_tensors = torch.zeros(batch_zh_ids.shape, dtype=torch.long)
        batch_zh_masks_tensors = batch_zh_masks_tensors.masked_fill(batch_zh_ids != 0, 1)
    else:
        batch_zh_ids, batch_zh_masks_tensors = None, None
    return batch_en_ids, batch_en_masks_tensors, \
           batch_zh_ids, batch_zh_masks_tensors


def get_accuracy(batch_logits, batch_labels):
    """ 计算一个batch的data的acc """
    with torch.no_grad():
        probabilities = F.softmax(batch_logits, dim=2)
        _, predictions = torch.max(probabilities.data, dim=2)
        # 计算acc
        total = batch_labels.size(0) * batch_labels.size(1)
        correct = (predictions == batch_labels).sum().item()
        acc = correct / total
        return acc


def train(model, dataloader, total_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    model.train()

    for epoch in range(args.num_epochs):
        total_loss, total_acc = 0, 0
        start_time = time.time()
        for index, data in enumerate(dataloader):
            en_ids, en_masks, zh_ids, zh_masks = [t.to(device) for t in data]
            optimizer.zero_grad()
            loss, logits = model(input_ids=en_ids,
                                 attention_mask=en_masks,
                                 decoder_input_ids=zh_ids,
                                 decoder_attention_mask=zh_masks,
                                 labels=zh_ids)[:2]

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            acc = get_accuracy(logits, zh_ids)
            total_acc += acc

            if (index + 1) % 500 == 0 or index == 0:
                # 显示本epoch中截至到此batch的平均loss 与 本batch的acc
                end_time = time.time()
                logging.info('[ Epoch {0}: {1}/{2}]'.format(epoch, index + 1, len(dataloader)))
                logging.info('  avg_loss: {0}, avg_acc: {1}, {2}s'.format(total_loss / (index + 1),
                                                                          round(acc, 3),
                                                                          round((end_time - start_time), 3)))
                start_time = end_time
        logging.info('\n *** In Epoch {0}, average loss: {1}, average acc: {2} *** \n'.format(
            epoch + 1,
            total_loss / len(dataloader),
            total_acc / len(dataloader)
        ))
        # 只保存参数
        model.save_pretrained(os.path.join(args.model_dir, 'model_epoch_{}'.format(epoch)))
    model.save_pretrained(args.model_dir)


def validate(model, dataloader):
    model.eval()

    for epoch in range(args.num_epochs):
        total_loss, total_acc = 0, 0
        for index, data in enumerate(dataloader):
            en_ids, en_masks, zh_ids, zh_masks = [t.to(device) for t in data]
            loss, logits = model(input_ids=en_ids,
                                 attention_mask=en_masks,
                                 decoder_input_ids=zh_ids,
                                 decoder_attention_mask=zh_masks,
                                 labels=zh_ids)[:2]

            total_loss += loss.item()

            acc = get_accuracy(logits, zh_ids)
            total_acc += acc

        logging.info(' *** Validation Result *** \nloss: {0} \nacc : {1}'.format(total_loss / len(dataloader),
                                                                                 total_acc / len(dataloader)))


def main():
    tokenizer = Tokenizer()
    logging.info('Using Device: {}'.format(device))

    if args.train:
        model_config = get_model_config()
        model = get_pretrained_model('train', model_config=model_config)
        model.to(device)
        train_dataset = datasets.Translate(mode='train',
                                           data_dir=args.data_dir,
                                           tokenizer=tokenizer,
                                           do_filter=False)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        total_steps = int(len(train_dataset) / args.train_batch_size * args.num_epochs)
        logging.info(
            "  Total training steps: {}".format(total_steps))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        train(model, train_dataloader, total_steps)
    if args.eval:
        model_config = get_model_config()
        model = get_pretrained_model('eval', model_config=model_config)
        model.to(device)
        eval_dataset = datasets.Translate(mode='eval',
                                          data_dir=args.data_dir,
                                          tokenizer=tokenizer,
                                          do_filter=False)

        logging.info("***** Running validating *****")
        logging.info("  Num examples = %d", len(eval_dataset))
        total_steps = int(len(eval_dataset) / args.train_batch_size)
        logging.info(
            "  Total training steps: {}".format(total_steps))

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=args.eval_batch_size,
                                     collate_fn=collate_fn)
        validate(model, eval_dataloader)


# def single_translate(english_text):
#     """ 只输入一句话，即batch_size==1 """
#     conf = configuration.Config()
#     tokenizer = tokenization.FullTokenizer(en_vocab_file=os.path.join(conf.file_config.data_path,
#                                                                       conf.file_config.en_vocab),
#                                            zh_vocab_file=os.path.join(conf.file_config.data_path,
#                                                                       conf.file_config.zh_vocab))
#     conf.model_config.src_vocab_size = tokenizer.get_en_vocab_size() + 2
#     conf.model_config.trg_vocab_size = tokenizer.get_zh_vocab_size() + 2
#     model = models.Transformer(conf)
#     model.to(device)
#     # model.load_state_dict(torch.load(os.path.join(conf.train_config.model_dir,
#     #                                               conf.train_config.model_name + '.pth')))
#     checkpoint = torch.load(os.path.join(conf.train_config.model_dir,
#                                          conf.train_config.model_name + '_epoch_{}.tar'.format(28)))
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
#
#     translate_dataset = datasets.Translate(mode='single_translate',
#                                            config=conf,
#                                            tokenizer=tokenizer,
#                                            texts=english_text)
#     translate_dataloader = DataLoader(translate_dataset,
#                                       batch_size=1,
#                                       collate_fn=collate_fn)
#     data = next(iter(translate_dataloader))
#     en_ids, _ = [t.to(device) if t is not None else t for t in data]         # [1, en_seq_len]
#     # decoder的初始输出为<BOS>
#     decoder_input = torch.tensor([tokenizer.get_zh_vocab_size()]).view(1, 1)   # [1, 1]
#
#     for i in range(51):
#         prediction_logits = model(en_ids, decoder_input)        # [1, i+1, vocab_size]
#         # 取出最后一个distribution，做argmax得到预测的新字
#         predictions = prediction_logits[:, -1, :]              # [batch_size, vocab_size]
#         predictions = F.softmax(predictions, dim=-1)
#         predict_zh_ids = torch.argmax(predictions, dim=-1)      # [batch_size]
#
#         # 若预测出的结果是<EOS>，则结束
#         if predict_zh_ids.data == tokenizer.get_zh_vocab_size() + 1:
#             break
#         # 否则，预测出的结果与先前的结果拼接，重新循环
#         else:
#             decoder_input = torch.cat([decoder_input, predict_zh_ids.view(1, 1)], dim=1)
#
#     # 将生成的中文id转回中文文字
#     translated_text = tokenizer.convert_zh_ids_to_text(list(decoder_input.detach().numpy()[0])[1: 51])
#     print('原文：', english_text)
#     print('翻译：', translated_text)
#     return translated_text


if __name__ == '__main__':
    main()
    # while True:
    #     english = input('English Text: ')
    #     _ = single_translate(english)
