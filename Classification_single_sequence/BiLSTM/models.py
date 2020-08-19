import torch.nn as nn
import torch


class TextLstm(nn.Module):
    """ 用于文本分类的LSTM模型 """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Sequential(
            nn.Embedding(config.model_config.vocab_size, config.model_config.emb_size),
            nn.Dropout(config.model_config.embedding_drop_prob)
        )
        self.lstm = nn.LSTM(
            input_size=config.model_config.emb_size,
            hidden_size=config.model_config.hidden_size,
            num_layers=config.model_config.num_of_layers,
            bidirectional=config.model_config.bidirectional,         # 双向
            dropout=config.model_config.drop_prob,
            batch_first=True                                         # 调整维度
        )
        self.num_of_directions = 2 if config.model_config.bidirectional is True else 1
        self.linear = nn.Sequential(
            nn.Linear(config.model_config.hidden_size * self.num_of_directions,
                      config.model_config.num_of_classes)
        )

    def forward(self, input_ids):

        # [batch_size, seq_len] -> [batch_size, seq_len, emb_size]
        word_embedding = self.embedding(input_ids)

        # torch中，若设置了双向网络，则输出已自动在hidden_size维度上进行了拼接
        # [batch_size, seq_len, emb_size] -> [batch_size, seq_len, hidden_size * num_of_directions]
        lstm_out, hidden = self.lstm(word_embedding)

        # [batch_size, seq_len, hidden_size * directions] -> [batch_size, hidden_size * directions]
        if self.config.model_config.bidirectional is True:
            # 若为双向网络，则手动拼接前后向的输出
            forward_last_output = lstm_out[:, -1, :self.config.model_config.hidden_size]
            backward_last_output = lstm_out[:, 0, self.config.model_config.hidden_size:]
            lstm_out = torch.cat((forward_last_output, backward_last_output), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        # lstm_out = lstm_out.squeeze(1)

        # [batch_size, hidden_size * directions -> [batch_size, num_classes]
        return self.linear(lstm_out)


if __name__ == '__main__':
    x = torch.rand(10, 24, 100)

    lstm = nn.LSTM(100, 128, num_layers=2, bidirectional=False, batch_first=True)
    linear = nn.Sequential(
        nn.Linear(256, 2),
        nn.Sigmoid()
    )

    output, _ = lstm(x)
    print(output.size())
    a = output[:, -1, :]
    print(a.size())
    print(a.squeeze(1).size())

    # forward = output[:, -1, :128]
    # backward = output[:, 0, 128:]
    # print(forward.size())
    # out = torch.cat((forward, backward), dim=1)
    # print(out.size())

    # lstm_out = output.contiguous().view(-1, 128 * 2)
    # print(lstm_out.shape)
    # output = linear(output)
    #
    # print(output.size())

    # back_last_output = output[:, 0, -128:]
    # print(back_last_output.shape)
    # forward_last_output = output[:, -1, :128]
    # print(forward_last_output.shape)
    # final_output = torch.cat((forward_last_output, back_last_output), dim=1)
    # print(final_output.shape)
    # print(final_output == output[:, 0, :])
    #
    # print(final_output == output[:, -1, :])
