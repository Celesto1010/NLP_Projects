import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader

import time
import logging
import datasets
import tokenization
import models
import configuration
import os
import argparse

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='variables for torch classification')
parser.add_argument('--mode', type=str, default='predict', help='run model training or run prediction')
parser.add_argument('--train', type=bool, default=False, help='if_training')
parser.add_argument('--eval', type=bool, default=True, help='if_validating')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_padding_mask(seq):
    """
    :param seq: 子词id形式，经过padding后的，一个batch的句子. [batch_size, seq_len]
    :return: 真实token位置标0，padding位置标1的新矩阵. [batch_size, 1, 1, seq_len]
    """
    masks = torch.zeros(seq.shape, dtype=torch.long)
    masks = masks.masked_fill(seq == 0, 1)
    return masks.unsqueeze(dim=1).unsqueeze(dim=2)


def create_look_ahead_mask(seq):
    """
    :param seq: 子词id形式，经过padding后的，一个batch的句子. [batch_size, seq_len]
    :return: 三角矩阵，对角线与左下方为0，右上方为1。即每一次只多看一个词，不看之后的词.   [seq_len, seq_len]
    """
    masks = torch.triu(torch.ones(seq.size(-1), seq.size(-1)), diagonal=1)
    return masks


def create_combined_mask(seq):
    """
    :param seq: 子词id形式，经过padding后的，一个batch的句子. [batch_size, seq_len]
    :return: 综合padding mask与look ahead mask
    """
    p_mask = create_padding_mask(seq)
    la_mask = create_look_ahead_mask(seq)
    combined_mask = torch.max(p_mask.long(), la_mask.long())
    return combined_mask


def collate_fn(batches):
    """ 每个batch做padding，而非所有样本做padding """
    batch_en_ids = [torch.tensor(batch[0]) for batch in batches]
    batch_en_ids = rnn_utils.pad_sequence(batch_en_ids, batch_first=True, padding_value=0)
    src_padding_mask = create_padding_mask(batch_en_ids)

    if batches[0][1] is not None:
        batch_zh_ids = [torch.tensor(batch[1]) for batch in batches]
        batch_zh_ids = rnn_utils.pad_sequence(batch_zh_ids, batch_first=True, padding_value=0)
        # trg_combined_mask用于喂入模型decoder，只使用 :-1 的字符做mask
        trg_combined_mask = create_combined_mask(batch_zh_ids[:, :-1])

        # loss mask, 用于将序列中不为零的地方置1，否则置0，搭配的是label，即使用 1: 的字符做mask
        loss_mask = torch.ones(batch_zh_ids[:, 1:].reshape(-1).shape, dtype=torch.long)
        loss_mask = loss_mask.masked_fill(batch_zh_ids[:, 1:].reshape(-1) == 0, 0)
    else:
        batch_zh_ids = None
        trg_combined_mask = None
        loss_mask = None
    return batch_en_ids, batch_zh_ids, src_padding_mask, trg_combined_mask, loss_mask


def score(logits, real_trg):
    probs = F.softmax(logits, dim=2)
    translated_ids = torch.argmax(probs, dim=2)

    # 计算acc
    total = real_trg.size(0) * real_trg.size(1)
    correct = (translated_ids == real_trg).sum().item()
    acc = correct / total
    return acc


def train(config, start_epoch, dataloader, optimizer, criterion, scheduler, total_lr, model):
    """ 模型的训练步骤 """
    model.train()
    # 总参数量与可训练参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(' *** start training, parameter total:{}, trainable:{} *** '.format(total, trainable))

    # Run Training
    for epoch in range(start_epoch, config.train_config.num_epochs):
        total_loss, total_acc = 0, 0
        start_time = time.time()
        for i, data in enumerate(dataloader):
            en_ids, zh_ids, src_padding_mask, trg_combined_mask, loss_mask = [t.to(device) for t in data]
            label_zh_ids = zh_ids[:, :-1]
            trg_zh_ids = zh_ids[:, 1:]

            optimizer.zero_grad()
            logits = model.forward(en_ids, label_zh_ids, src_padding_mask, trg_combined_mask)           # logits: [batch_size, trg_len, trg_vocab_size]
            loss_logits = logits.reshape(-1, config.model_config.trg_vocab_size)                        # -> [batch_size * trg_len, trg_vocab_size]
            loss = criterion(loss_logits, trg_zh_ids.reshape(-1))                                       # trg: -> [batch_size * trg_len]

            loss *= loss_mask
            loss = torch.mean(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train_config.max_grad_norm)
            optimizer.step()

            total_lr += scheduler.get_lr()[0]
            scheduler.step()

            acc = score(logits, trg_zh_ids)
            total_acc += acc
            total_loss += loss.item()

            if (i + 1) % 500 == 0 or i == 0:
                end_time = time.time()
                # 显示历史平均lr、本epoch中截至到此batch的平均loss 与 此batch的的bleu
                logging.info('[ Epoch {0}: {1}/{2} ]'.format(epoch + 1, i + 1, len(dataloader)))
                logging.info('  avg_loss: {0}, acc: {3}, avg_lr: {2}, {1}s'.format(
                    total_loss / (i + 1),
                    round((end_time - start_time), 3),
                    round(total_lr / (i + 1 + epoch * len(dataloader)), 6),
                    round(acc, 3))
                )
                start_time = end_time
        logging.info('\n *** In Epoch {0}, average loss: {1}, average acc: {2} *** \n'.format(
            epoch + 1,
            total_loss / len(dataloader),
            total_acc / len(dataloader)
        ))

        # 每个epoch结束后保存一个checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
            'total_lr': total_lr
        }, os.path.join(config.train_config.model_dir,
                        config.train_config.model_name + '_epoch_{}.tar'.format(epoch + 1)))
    # 只保存参数
    torch.save(model.state_dict(), os.path.join(config.train_config.model_dir, config.train_config.model_name + '.pth'))


def validate(config, dataloader, criterion, model):
    if args.train:
        logging.info(' *** Run validating after training. *** ')
    else:
        checkpoint = torch.load(os.path.join(config.train_config.model_dir,
                                             config.train_config.model_name + '_epoch_{}.tar'.format(50)),
                                map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # model.load_state_dict(torch.load(os.path.join(config.train_config.model_dir,
        #                                               config.train_config.model_name + '.pth'),
        #                       map_location=device))

    model.eval()
    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, data in enumerate(dataloader):
            en_ids, zh_ids, src_padding_mask, trg_combined_mask, loss_mask = [t.to(device) for t in data]
            label_zh_ids = zh_ids[:, :-1]
            trg_zh_ids = zh_ids[:, 1:]

            logits = model.forward(en_ids, label_zh_ids, src_padding_mask, trg_combined_mask)
            loss_logits = logits.reshape(-1, config.model_config.trg_vocab_size)
            loss = criterion(loss_logits, trg_zh_ids.reshape(-1))
            loss *= loss_mask
            loss = torch.mean(loss)

            acc = score(logits, trg_zh_ids)
            total_acc += acc
            total_loss += loss.item()

        logging.info(' *** Validation Result *** \nloss: {0} \nacc : {1}'.format(total_loss / len(dataloader),
                                                                                 total_acc / len(dataloader)))
        with open('eval_result.txt', 'w', encoding='utf-8') as f:
            f.write('Total Loss: {0} \nTotal Acc: {1}'.format(total_loss / len(dataloader), total_acc / len(dataloader)))


def run(config, dataloader, model, mode, start_epoch=None, total_steps=None):
    """ 初始化或加载optimizer、criterion、scheduler与model，准备执行training或validating """
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(reduction='none')

    warmup_step = int(total_steps * config.train_config.warmup_var)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                  base_lr=config.train_config.base_lr,
                                                  max_lr=config.train_config.max_lr,
                                                  step_size_up=warmup_step,
                                                  step_size_down=total_steps,
                                                  cycle_momentum=False)
    if mode == 'train':
        logging.info(' *** Train from the beginning. *** ')
        train(config,
              start_epoch=0,
              dataloader=dataloader,
              optimizer=optimizer,
              criterion=criterion,
              scheduler=scheduler,
              total_lr=0,
              model=model)

    elif mode == 'continue_train':
        checkpoint = torch.load(os.path.join(config.train_config.model_dir,
                                             config.train_config.model_name + '_epoch_{}.tar'.format(start_epoch)))
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        total_lr = checkpoint['total_lr']

        logging.info(' *** Latest Checkpoint Loaded. Continue Training From Epoch {} *** '.format(start_epoch))
        train(config,
              start_epoch=start_epoch,
              dataloader=dataloader,
              optimizer=optimizer,
              criterion=criterion,
              scheduler=scheduler,
              total_lr=total_lr,
              model=model)

    elif mode == 'eval':
        logging.info(' *** Validating *** ')
        validate(config, dataloader, criterion, model)


def main():
    conf = configuration.Config()
    tokenizer = tokenization.FullTokenizer(en_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.en_vocab),
                                           zh_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.zh_vocab))
    logging.info('Using Device: {}'.format(device))
    conf.model_config.src_vocab_size = tokenizer.get_en_vocab_size() + 2
    conf.model_config.trg_vocab_size = tokenizer.get_zh_vocab_size() + 2
    model = models.Transformer(conf)
    model = model.to(device)

    if args.train:
        train_dataset = datasets.Translate(mode='train',
                                           config=conf,
                                           tokenizer=tokenizer,
                                           auto_padding=conf.train_config.auto_padding,
                                           do_filter=False)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Total training steps: {}".format(train_dataset.num_steps))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=conf.train_config.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        run(config=conf,
            dataloader=train_dataloader,
            model=model,
            mode='train',
            start_epoch=0,
            total_steps=train_dataset.num_steps)

    if args.eval:
        eval_dataset = datasets.Translate(mode='eval',
                                          config=conf,
                                          tokenizer=tokenizer,
                                          auto_padding=conf.train_config.auto_padding,
                                          do_filter=False)

        logging.info("***** Running validating *****")
        logging.info("  Num examples = %d", len(eval_dataset))
        logging.info("  Total validating steps: {}".format(eval_dataset.num_steps))

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=conf.train_config.eval_batch_size,
                                     collate_fn=collate_fn)
        run(config=conf,
            dataloader=eval_dataloader,
            model=model,
            mode='eval',
            start_epoch=0,
            total_steps=eval_dataset.num_steps)


def single_translate(english_text):
    """ 只输入一句话，即batch_size==1 """
    conf = configuration.Config()
    tokenizer = tokenization.FullTokenizer(en_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.en_vocab),
                                           zh_vocab_file=os.path.join(conf.file_config.data_path,
                                                                      conf.file_config.zh_vocab))
    conf.model_config.src_vocab_size = tokenizer.get_en_vocab_size() + 2
    conf.model_config.trg_vocab_size = tokenizer.get_zh_vocab_size() + 2
    model = models.Transformer(conf)
    model.to(device)

    translate_dataset = datasets.Translate(mode='single_translate',
                                           config=conf,
                                           tokenizer=tokenizer,
                                           texts=english_text)
    # encoder输入即为原始英文id
    en_ids, _ = translate_dataset[0]
    en_ids = torch.tensor(en_ids).unsqueeze(dim=0)
    # decoder的初始输出为<BOS>
    decoder_input = torch.tensor([tokenizer.get_zh_vocab_size()]).view(1, 1)   # [1, 1]

    model.load_state_dict(torch.load(os.path.join(conf.train_config.model_dir,
                                                  conf.train_config.model_name + '.pth'),
                                     map_location=device))
    # checkpoint = torch.load(os.path.join(conf.train_config.model_dir,
    #                                      conf.train_config.model_name + '_epoch_{}.tar'.format(50)),
    #                         map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for i in range(51):
        if torch.cuda.is_available():
            prediction_logits = model(en_ids.cuda(),
                                      decoder_input.cuda())        # [1, i+1, vocab_size]
        else:
            prediction_logits = model(en_ids,
                                      decoder_input)
        # 取出最后一个distribution，做argmax得到预测的新字
        predictions = prediction_logits[:, -1, :]              # [batch_size, vocab_size]
        predictions = F.softmax(predictions, dim=-1)
        predict_zh_ids = torch.argmax(predictions, dim=-1)      # [batch_size]

        # 若预测出的结果是<EOS>，则结束
        if predict_zh_ids.data == tokenizer.get_zh_vocab_size() + 1:
            break
        # 否则，预测出的结果与先前的结果拼接，重新循环
        else:
            if torch.cuda.is_available():
                decoder_input = torch.cat([decoder_input.cuda(), predict_zh_ids.view(1, 1)], dim=1)
            else:
                decoder_input = torch.cat([decoder_input, predict_zh_ids.view(1, 1)], dim=1)

    # 将生成的中文id转回中文文字
    translated_text = tokenizer.convert_zh_ids_to_text(list(decoder_input.cpu().detach().numpy()[0])[1: 51])
    print('原文：', english_text)
    print('翻译：', translated_text)
    return translated_text


if __name__ == '__main__':
    if args.mode == 'train':
        main()
    elif args.mode == 'predict':
        while True:
            english = input('English Text: ')
            _ = single_translate(english)


