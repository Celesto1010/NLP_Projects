import transformers
import datasets
import logging
from torch.utils.data import DataLoader
import torch
import os
from datetime import datetime
import argparse

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='variables for torch classification')
parser.add_argument('--mode', type=str, default='predict', help='run model training or run prediction')
parser.add_argument('--train', type=bool, default=True, help='if_training')
parser.add_argument('--eval', type=bool, default=True, help='if_validating')
parser.add_argument('--train_batch_size', default=20, type=int, required=True, help='训练batch size')
parser.add_argument('--num_epoch', default=10, type=int, help='训练epoch')
parser.add_argument('--stride', default=768, type=int, help='sample步长')
parser.add_argument('--min_length', default=0, type=int, help='限制每篇被选取的文章的最短长度')
parser.add_argument('--lr', default=1.5e-4, type=float, help='基准学习率')
parser.add_argument('--warmup_step', default=2000, type=int, help='warmup步数')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='限制梯度最大值')
parser.add_argument('--data_dir', default='./data', type=str, help='训练数据保存文件夹')
parser.add_argument('--output_dir', default='./models', type=str, help='输出模型的保存路径')
parser.add_argument('--pretrained_model_dir', default=r'E:\NLP_Projects\Models\GPT2\GPT2',
                    type=str, help='预训练模型文件路径')
parser.add_argument('--model_from_pretrained', type=bool, default=False, help='是否加载预训练文件')
parser.add_argument('--pretrained_epoch', type=int, default=15, help='预训练模型的轮数')
parser.add_argument('--tokenizer_vocab', default='./vocab/vocab_small.txt', type=str, help='词典文件')
parser.add_argument('--bert_tokenizer_dir', default=r'E:\NLP_Projects\Models\Bert_Pretrained\Chinese\Bert_Tokenizer',
                    type=str, help='BertTokenizer, 21128字典的文件夹路径')
args = parser.parse_args()


def get_model_config():
    """ 加载gpt2模型配置。已下载好，在config文件夹中 """
    model_config_path = 'config/model_config_small.json'
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(model_config_path)
    # print('config:\n' + model_config.to_json_string())
    return model_config


def get_tokenizer(file_config=None, use_vocab=False):
    """ 加载tokenizer
        use_vocab: 是否用现有的字典文件"""
    if use_vocab:
        tokenizer = transformers.BertTokenizer(vocab_file=args.tokenizer_vocab)
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained(file_config.bert_tokenizer_dir)
    logging.info('字典大小：{}'.format(len(tokenizer)))
    return tokenizer


def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('using device: {}'.format(device))
    return device


def get_model(model_config):
    if not args.model_from_pretrained:
        # 从零训练
        logging.info(' --- Run Training From Beginning --- ')
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        # 基于预训练模型训练
        logging.info(' --- Run Training From Epoch {} --- '.format(args.pretrained_epoch))
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(
            os.path.join(args.pretrained_model_dir,
                         'model_epoch_{}'.format(args.pretrained_epoch)))
    return model


def train(dataloader, model, device, total_steps=None):
    model.train()
    # 统计总参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(' *** start training, parameter total:{}, trainable:{} *** '.format(total, trainable))

    # 定义优化器与scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             args.warmup_step,
                                                             total_steps)

    logging.info(' ----------- Start Training --------------')

    for epoch in range(args.num_epoch):
        total_loss = 0

        for i, batch_inputs in enumerate(dataloader):
            optimizer.zero_grad()
            batch_inputs.to(device)
            outputs = model.forward(input_ids=batch_inputs,
                                    labels=batch_inputs)
            loss, logits = outputs[:2]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

            if (i + 1) % 500 == 0 or i == 0:
                logging.info('[ {4}: {5}, Epoch {0}: {1}/{2} AVG_LOSS: {3} ]'.format(
                    epoch + 1,
                    i + 1,
                    len(dataloader),
                    total_loss / (i + 1),
                    datetime.now().hour,
                    datetime.now().minute))

        logging.info('\n *** In Epoch {0}, average loss: {1} *** \n'.format(
            epoch + 1,
            total_loss / len(dataloader))
        )

        logging.info('Saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        epoch_model_dir = os.path.join(args.output_dir, 'model_epoch_{}'.format(epoch + 1 + args.pretrained_epoch))
        if not os.path.exists(epoch_model_dir):
            os.mkdir(epoch_model_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(epoch_model_dir)


def main():
    model_config = get_model_config()
    device = get_device()
    tokenizer = get_tokenizer(use_vocab=True)
    assert tokenizer.vocab_size == model_config.vocab_size
    model = get_model(model_config)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(' Trainable Parameters : {} *** '.format(trainable))

    if args.train:
        processor = datasets.PoetDataProcessor(data_dir=args.data_dir,
                                               mode='train')
        dataset = datasets.GPT2Dataset(data_processor=processor,
                                       mode='train',
                                       tokenizer=tokenizer,
                                       n_ctx=model_config.n_ctx,
                                       stride=args.stride,
                                       min_length=args.min_length)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(dataset))
        num_steps = int(len(dataset) * args.num_epoch / args.train_batch_size)
        logging.info("  Total training steps: {}".format(num_steps))

        train_dataloader = DataLoader(dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True)

        train(dataloader=train_dataloader,
              model=model,
              device=device,
              total_steps=num_steps)


if __name__ == '__main__':
    main()