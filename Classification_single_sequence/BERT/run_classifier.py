import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import time
import logging
import datasets
from transformers import BertForSequenceClassification, BertTokenizer
import configuration
from torch.utils.data import DataLoader
import os
import argparse

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description='variables for torch classification')
parser.add_argument('--mode', type=str, default='train', help='run model training or run prediction')
parser.add_argument('--train', type=bool, default=True, help='if_training')
parser.add_argument('--dev', type=bool, default=True, help='if_validating')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batches):
    """ 每个batch做padding，而非所有样本做padding """
    input_ids = [torch.tensor(batch[0]) for batch in batches]
    input_ids = rnn_utils.pad_sequence(input_ids, batch_first=True, padding_value=0)

    segment_ids = [torch.tensor(batch[1]) for batch in batches]
    segment_ids = rnn_utils.pad_sequence(segment_ids, batch_first=True, padding_value=0)

    # 在bert里，pad部分mask标0（与自己实现的transformer相反）
    masks_tensors = torch.zeros(input_ids.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(input_ids != 0, 1)

    if batches[0][2] is not None:
        labels = torch.tensor([batch[2] for batch in batches], dtype=torch.long)
    else:
        labels = None

    return input_ids, segment_ids, masks_tensors, labels


def get_predictions(batch_output, batch_labels=None, compute_acc=True):
    """ 计算一个batch的data的acc """
    correct = 0
    total = 0

    with torch.no_grad():
        if len(batch_output) == 2:
            logits = batch_output[1]
        else:
            logits = batch_output[0]
        probabilities = F.softmax(logits, dim=1)
        prob, predictions = torch.max(probabilities.data, 1)

        # 计算准确率
        if compute_acc:
            total += batch_labels.size(0)           # batch_size
            correct += (predictions == batch_labels).sum().item()
            acc = correct / total
            return prob, predictions, acc

        else:
            return prob, predictions


def run(config, dataloader, model, mode):
    """ 运行 """
    # Settings
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    if mode == 'train':
        logging.info(f"""
                整个分类器参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad)}
                线性分类器参数量：{sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)}
                """)

        num_batches = len(dataloader)

        # Run Training
        for epoch in range(config.num_epochs):
            total_loss, total_acc = 0, 0
            start_time = time.time()
            for index, data in enumerate(dataloader):
                tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

                optimizer.zero_grad()
                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)

                loss = outputs[0]
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                _, _, acc = get_predictions(outputs, labels)
                total_acc += acc

                if (index + 1) % 100 == 0:
                    # 显示本epoch中截至到此batch的平均loss 与 平均acc
                    end_time = time.time()
                    logging.info('[ Epoch {0}: {1}/{2}]'.format(epoch, index + 1, num_batches))
                    logging.info('  avg_loss: {0}, avg_acc: {1}, {2}s'.format(total_loss / (index + 1),
                                                                              round(total_acc / (index + 1), 3),
                                                                              round((end_time - start_time), 3)))
                    start_time = end_time

            # 只保存参数
            model.save_pretrained('./models/model_epoch_{}'.format(epoch))

    elif mode == 'eval':
        if args.train:
            logging.info(' *** Run validating after training. *** ')
        elif os.path.exists(os.path.join(config.model_dir, config.model_name)):
            model = BertForSequenceClassification.from_pretrained(config.model_dir)
            model = model.to(device)
            logging.info(' *** Pretrained model found. Use pretrained model to validate. *** ')
        else:
            logging.info(' *** No pretrained model. Stop validating. *** ')
            return
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for index, data in enumerate(dataloader):
                tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)
                loss = outputs[0]
                total_loss += loss.item()

                _, _, acc = get_predictions(outputs, labels)
                total_acc += acc

            logging.info(' *** Validation Result *** \nloss: {0} \nacc : {1}'.format(total_loss / len(dataloader),
                                                                                     total_acc / len(dataloader)))

    else:
        logging.info(' Error: No Such Mode. ')


def train():
    conf = configuration.Config()
    tokenizer = BertTokenizer.from_pretrained(conf.pretrained_model_name)

    # 加载bert的预训练模型。指定cache文件夹路径
    pretrained_model = os.path.join(conf.pretrained_model_path, conf.pretrained_model_name)
    if not os.path.exists(pretrained_model):
        os.mkdir(pretrained_model)
        model = BertForSequenceClassification.from_pretrained(conf.pretrained_model_name, num_labels=conf.num_labels,
                                                              cache_dir=os.path.join(pretrained_model, './cache'))
        model.save_pretrained(pretrained_model)
    else:
        model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=conf.num_labels)
    model = model.to(device)

    if args.train:
        model.train()
        train_dataset = datasets.OnlineShopping(mode='train',
                                                config=conf,
                                                tokenizer=tokenizer,
                                                auto_padding=conf.auto_padding)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Total training steps: {}".format(train_dataset.num_steps))

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=conf.train_batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn)

        run(config=conf, dataloader=train_dataloader, model=model, mode='train')

    if args.dev:
        model.eval()
        dev_dataset = datasets.OnlineShopping(mode='dev',
                                              config=conf,
                                              tokenizer=tokenizer,
                                              auto_padding=conf.auto_padding)

        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(dev_dataset))
        logging.info("  Total training steps: {}".format(dev_dataset.num_steps))

        dev_dataloader = DataLoader(dev_dataset,
                                    batch_size=conf.dev_batch_size,
                                    shuffle=True,
                                    collate_fn=collate_fn)

        run(config=conf, dataloader=dev_dataloader, model=model, mode='eval')


def predict(texts):
    conf = configuration.Config()
    tokenizer = BertTokenizer.from_pretrained(conf.pretrained_model_name)
    model = BertForSequenceClassification.from_pretrained(conf.pretrained_model_name, num_labels=conf.num_labels)
    model = model.to(device)

    if os.path.exists(os.path.join(conf.model_dir, conf.model_name)):
        model.load_state_dict(torch.load(os.path.join(conf.model_dir,
                                                      conf.model_name)))
    else:
        logging.info(' *** No model available. *** ')
        return

    predict_dataset = datasets.OnlineShopping(mode='single_predict',
                                              config=conf,
                                              tokenizer=tokenizer,
                                              auto_padding=True,
                                              texts=texts)

    predict_dataloader = DataLoader(predict_dataset,
                                    batch_size=len(texts),
                                    collate_fn=collate_fn)
    data = next(iter(predict_dataloader))
    tokens_tensors, segments_tensors, masks_tensors, _ = [t.to(device) if t is not None else t for t in data]
    outputs = model(input_ids=tokens_tensors,
                    token_type_ids=segments_tensors,
                    attention_mask=masks_tensors)
    print(outputs)
    probs, predictions = get_predictions(outputs, compute_acc=False)
    return dict(zip(texts, [{'result': label, 'probability': prob} for label, prob in
                            zip([predict_dataset.convert_label_id_to_value(prediction.item()) for prediction in
                                 predictions],
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
