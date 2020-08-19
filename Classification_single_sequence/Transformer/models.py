import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.model_config.vocab_size, config.model_config.dim)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.model_config.encoder_layers)])
        self.dropout = nn.Dropout(config.model_config.drop_rate)
        self.linear = nn.Linear(config.model_config.dim, config.model_config.num_classes)

    def forward(self, input_ids):
        input_padding_masks = self.create_padding_mask(input_ids)

        # 1. embedding and positional embedding
        input_emb = self.embedding(input_ids)
        input_emb = self.positional_encoding(input_emb, input_emb.size(1))
        x = self.dropout(input_emb)

        # 2. encoder
        for i, enc_layer in enumerate(self.encoder):
            x, _ = enc_layer(x, input_padding_masks)

        # 3. dense
        encoder_output = x.mean(dim=1)
        logits = self.linear(encoder_output)
        return logits

    @staticmethod
    def create_padding_mask(seq):
        """
        :param seq: 子词id形式，经过padding后的，一个batch的句子. [batch_size, seq_len]
        :return: 真实token位置标0，padding位置标1的新矩阵. [batch_size, 1, 1, seq_len]
        """
        masks = torch.zeros(seq.shape, dtype=torch.long)
        masks = masks.masked_fill(seq == 0, 1)
        return masks.unsqueeze(dim=1).unsqueeze(dim=2)

    def positional_encoding(self, input_emb, seq_len):
        pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / self.config.model_config.dim))
                            for i in range(self.config.model_config.dim)]
                           for pos in range(seq_len)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        output = input_emb + nn.Parameter(pe, requires_grad=False)
        return output


class EncoderLayer(nn.Module):
    """ 一个Encoder层，包括多头注意力子层、FeedForward子层与残差连接 """
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.layer_norm1 = nn.LayerNorm(config.model_config.dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(config.model_config.dim, eps=1e-6)

        self.dropout1 = nn.Dropout(config.model_config.drop_rate)
        self.dropout2 = nn.Dropout(config.model_config.drop_rate)

    def forward(self, input_emb, input_mask):
        # 1. Multihead Attention sub-layer
        # 对于encoder，qkv都是input_emb自己
        att_output, attn = self.mha(input_emb, input_emb, input_emb, input_mask)
        att_output = self.dropout1(att_output)
        out1 = self.layer_norm1(input_emb + att_output)

        # 2. Feed-forward sub-layer
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layer_norm2(out1 + ffn_out)

        return out2, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_heads = config.model_config.num_heads          # 头数
        self.d_model = config.model_config.dim                  # 整体维度
        assert self.d_model % self.num_heads == 0
        self.sub_head_dim = self.d_model // self.num_heads      # 每个子头的维度

        self.Wq = nn.Linear(config.model_config.dim, config.model_config.dim)
        self.Wk = nn.Linear(config.model_config.dim, config.model_config.dim)
        self.Wv = nn.Linear(config.model_config.dim, config.model_config.dim)

        self.dense = nn.Linear(config.model_config.dim, config.model_config.dim)

    def split_heads(self, mat):
        batch_size = mat.size(0)
        # 把矩阵维度分成num_heads个sub_head_dim的维度. [batch_size, seq_len, num_heads, sub_head_dim]
        mat = mat.view(batch_size, -1, self.num_heads, self.sub_head_dim)
        # 转变维度. [batch_size, num_heads, seq_len, sub_head_dim]
        mat = torch.transpose(mat, 1, 2)
        # mat = mat.permute(0, 2, 1, 3)    用permute和transpose效果一样
        return mat

    def forward(self, q, k, v, input_mask):
        """
        Self Attention的计算过程
        :param q: Query查询矩阵.                       [batch_size, seq_len, dim]
        :param k: Keys键值矩阵。Q与K相匹配，计算匹配度     [batch_size, seq_len, dim]
        :param v: Value值矩阵。                       [batch_size, seq_len, dim]
        :param input_mask: Mask Tensor. [batch_size, 1, 1, seq_len]
        :return: output: self attention层的输出. [batch_size, seq_len, dim]
                 attention_weights: 代表在每个头中，每个词对包括自己在内的其他词的注意权重. [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size = q.size(0)

        # [batch_size, seq_len, dim]
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        # [batch_size, num_heads, seq_len, sub_head_dim]
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # [batch_size, num_heads, seq_len, seq_len]
        matmul_qk = torch.matmul(Q, torch.transpose(K, -1, -2))
        scaled_attention_logits = matmul_qk / (K.size(-1) ** 0.5)
        # 带上mask，正常token位置+0，mask位置变为负极大，保证softmax后，mask处位置变成0，即不再注意
        if input_mask is not None:
            scaled_attention_logits += (input_mask * -1e9)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # [batch_size, num_heads, seq_len, sub_head_dim]
        scaled_attention = torch.matmul(attention_weights, V)

        # 转回原来的维度 [batch_size, seq_len, num_heads, sub_head_dim]
        scaled_attention = torch.transpose(scaled_attention, 1, 2)
        concat_attention = scaled_attention.reshape([batch_size, -1, self.d_model])

        output = self.dense(concat_attention)
        return output, attention_weights


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.model_config.dim, config.model_config.ff_size),
            nn.ReLU(),
            nn.Linear(config.model_config.ff_size, config.model_config.dim)
        )

    def forward(self, input_tensor):
        return self.ffn(input_tensor)


if __name__ == '__main__':
    import configuration
    conf = configuration.Config()

    # batch_size = 3, seq_len = 5
    input_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                              [2, 3, 4, 5, 6, 7, 0, 0]])

    def create_look_ahead_mask(seq):
        """
        :param seq: 子词id形式，经过padding后的，一个batch的句子. [batch_size, seq_len]
        :return: 三角矩阵，对角线与左下方为0，右上方为1。即每一次只多看一个词，不看之后的词.   [seq_len, seq_len]
        """
        masks = torch.triu(torch.ones(seq.size(-1), seq.size(-1)), diagonal=1)
        return masks

    enc = TransformerEncoder(conf)
    out = enc(input_seq)
    print(out)
    print(out.shape)
