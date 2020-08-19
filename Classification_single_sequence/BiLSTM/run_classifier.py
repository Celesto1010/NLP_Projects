import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import time
import logging
import datasets
import tokenization
import models
import configuration
from torch.utils.data import DataLoader
import os
import argparse

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='variables for torch classification')
parser.add_argument('--mode', type=str, default='predict', help='run model training or run prediction')
parser.add_argument('--train', type=bool, default=False, help='if_training')
parser.add_argument('--dev', type=bool, default=True, help='if_validating')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batches):
    """ 每个batch做padding，而非所有样本做padding """
    batch_ids = [torch.tensor(batch[0]) for batch in batches]
    batch_ids = rnn_utils.pad_sequence(batch_ids, batch_first=True, padding_value=0)

    if batches[0][1] is not None:
        batch_labels = torch.tensor([batch[1] for batch in batches], dtype=torch.long)
    else:
        batch_labels = None
    return batch_ids, batch_labels


def get_predictions(logits, labels=None, compute_acc=False):
    correct = 0
    total = 0

    with torch.no_grad():
        probabilities = F.softmax(logits, dim=1)
        prob, predictions = torch.max(probabilities.data, 1)

        # 计算准确率
        if compute_acc:
            total += labels.size(0)  # batch_size
            correct += (predictions == labels).sum().item()
            acc = correct / total
            return prob, predictions, acc

        else:
            return prob, predictions


def run(config, dataloader, model, mode, total_steps=None):
    """ 运行 """
    # Settings
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    num_batches = len(dataloader)

    if mode == 'train':
        if os.path.exists(os.path.join(config.train_config.model_dir, config.train_config.model_name)):
            model.load_state_dict(torch.load(os.path.join(config.train_config.model_dir,
                                                          config.train_config.model_name)))
            logging.info(' *** Pretrained model detected. Doing transfer learning. *** ')
        else:
            logging.info(' *** No pretrained model detected. Train from the beginning. *** ')
        # 总参数量与可训练参数量
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(' *** start training, parameter total:{}, trainable:{} *** '.format(total, trainable))

        # 学习率调整策略
        warmup_step = int(total_steps * config.train_config.warmup_var)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=config.train_config.base_lr,
                                                      max_lr=config.train_config.max_lr,
                                                      step_size_up=warmup_step,
                                                      step_size_down=total_steps,
                                                      cycle_momentum=False)

        # Run Training
        total_lr = 0
        for epoch in range(config.train_config.num_epochs):
            total_loss, total_acc = 0, 0
            start_time = time.time()
            for i, (text_ids, labels) in enumerate(dataloader):
                text_ids = text_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model.forward(text_ids)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train_config.max_grad_norm)
                optimizer.step()

                total_lr += scheduler.get_lr()[0]
                scheduler.step()

                _, _, acc = get_predictions(logits, labels, True)
                total_acc += acc
                total_loss += loss.item()

                if (i + 1) % 100 == 0:
                    end_time = time.time()
                    # 显示历史平均lr、本epoch中截至到此batch的平均loss 与 此batch的的acc
                    logging.info('[ Epoch {0}: {1}/{2} ]'.format(epoch + 1, i + 1, num_batches))
                    logging.info('  avg_loss: {0}, acc: {1}, avg_lr: {3}, {2}s'.format(
                        total_loss / (i + 1),
                        round(acc, 3),
                        round((end_time - start_time), 3),
                        round(total_lr / (i + 1 + epoch * num_batches), 6))
                                 )
                    start_time = end_time
            logging.info('\n *** In Epoch {0}, average loss: {1}, average acc: {2} *** \n'.format(
                epoch + 1,
                total_loss / num_batches,
                round(total_acc / num_batches, 4))
                         )

        # 只保存参数
        torch.save(model.state_dict(), os.path.join(config.train_config.model_dir, config.train_config.model_name))

    elif mode == 'eval':
        if args.train:
            logging.info(' *** Run validating after training. *** ')
        elif os.path.exists(os.path.join(config.train_config.model_dir, config.train_config.model_name)):
            model.load_state_dict(torch.load(os.path.join(config.train_config.model_dir,
                                                          config.train_config.model_name)))
            logging.info(' *** Pretrained model found. Use pretrained model to validate. *** ')
        else:
            logging.info(' *** No pretrained model. Stop validating. *** ')
            return
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (text_ids, labels) in enumerate(dataloader):
                text_ids = text_ids.to(device)
                labels = labels.to(device)

                logits = model(text_ids)
                loss = criterion(logits, labels)

                _, _, acc = get_predictions(logits, labels, True)
                total_acc += acc
                total_loss += loss.item()

            logging.info(' *** Validation Result *** \nloss: {0} \nacc : {1}'.format(total_loss / num_batches,
                                                                                     total_acc / num_batches))
            with open('eval_result.txt', 'w', encoding='utf-8') as f:
                f.write('Total Loss: {0} \nTotal Acc: {1}'.format(total_loss / num_batches, total_acc / num_batches))

    else:
        logging.info(' Error: No Such Mode. ')


def train():
    conf = configuration.LstmConfig(vocab_size=68355)
    tokenizer = tokenization.FullTokenizer(vocab_file=conf.file_config.vocab_file)

    model = models.TextLstm(conf)
    model = model.to(device)

    if args.train:
        model.train()
        train_dataset = datasets.OnlineShopping(mode='train',
                                                config=conf,
                                                tokenizer=tokenizer,
                                                auto_padding=conf.train_config.auto_padding)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Total training steps: {}".format(train_dataset.num_steps))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=conf.train_config.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        run(config=conf, dataloader=train_dataloader, model=model, mode='train', total_steps=train_dataset.num_steps)

    if args.dev:
        model.eval()
        dev_dataset = datasets.OnlineShopping(mode='dev',
                                              config=conf,
                                              tokenizer=tokenizer,
                                              auto_padding=conf.train_config.auto_padding)

        logging.info("***** Running validating *****")
        logging.info("  Num examples = %d", len(dev_dataset))
        logging.info("  Total validating steps: {}".format(dev_dataset.num_steps))

        train_dataloader = DataLoader(dev_dataset,
                                      batch_size=conf.train_config.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        run(config=conf, dataloader=train_dataloader, model=model, mode='eval')


def predict(texts):
    conf = configuration.LstmConfig(vocab_size=68355)
    tokenizer = tokenization.FullTokenizer(vocab_file=conf.file_config.vocab_file)

    model = models.TextLstm(conf)
    model = model.to(device)

    if os.path.exists(os.path.join(conf.train_config.model_dir, conf.train_config.model_name)):
        logging.info(' *** Loading model ***')
        model.load_state_dict(torch.load(os.path.join(conf.train_config.model_dir,
                                                      conf.train_config.model_name)))
    else:
        logging.info(' *** No model available. *** ')
        return

    predict_dataset = datasets.OnlineShopping(mode='single_predict',
                                              config=conf,
                                              tokenizer=tokenizer,
                                              auto_padding=True,
                                              texts=texts)
    predict_dataloader = DataLoader(predict_dataset,
                                    batch_size=len(predict_dataset),
                                    collate_fn=collate_fn)
    data = next(iter(predict_dataloader))
    text_ids, _ = [t.to(device) if t is not None else t for t in data]
    logits = model(text_ids)
    print(logits)
    probs, predictions = get_predictions(logits)

    return dict(zip(texts, [{'result': label, 'probability': prob} for label, prob in
                            zip([predict_dataset.convert_label_id_to_value(prediction.item()) for prediction in predictions],
                                [prob.item() for prob in probs])]))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        texts = ['今天是个好天气，真不错',
                 '昨天被老板骂了一顿，哼！']
        results = predict(texts)
        for k, v in results.items():
            print(k, v)
