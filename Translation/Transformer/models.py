from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module, ABC):
    """ Implement the PE function. """
    def __init__(self, config, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, config.model_config.dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.model_config.dim, 2) *
                             -(math.log(10000.0) / config.model_config.dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_emb):
        input_emb = input_emb + nn.Parameter(self.pe[:, :input_emb.size(1)], requires_grad=False)
        return input_emb


class Transformer(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.linear = nn.Linear(config.model_config.dim, config.model_config.trg_vocab_size)

    def forward(self, src_input_ids, trg_input_ids, src_padding_mask=None, trg_combined_mask=None):
        enc_output = self.encoder(src_input_ids, src_padding_mask)

        dec_output, _ = self.decoder(trg_input_ids, enc_output, src_padding_mask, trg_combined_mask)
        logits = self.linear(dec_output)
        return logits


class TransformerEncoder(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.model_config.src_vocab_size, config.model_config.dim)
        self.positional_encoding = PositionalEncoding(config)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.model_config.encoder_layers)])
        self.dropout = nn.Dropout(config.model_config.drop_rate)

    def forward(self, src_input_ids, input_padding_mask):

        # 1. embedding and positional embedding
        input_emb = self.embedding(src_input_ids)
        # input_emb = positional_encoding(self.config, input_emb, input_emb.size(1))
        input_emb = self.positional_encoding(input_emb)
        x = self.dropout(input_emb)

        # 2. encoder
        for i, enc_layer in enumerate(self.encoder):
            x, attn = enc_layer(x, input_padding_mask)

        return x


class EncoderLayer(nn.Module, ABC):
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


class TransformerDecoder(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.model_config.trg_vocab_size, config.model_config.dim)
        self.positional_encoding = PositionalEncoding(config)
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.model_config.decoder_layers)])
        self.dropout = nn.Dropout(config.model_config.drop_rate)

    def forward(self, trg_input_ids, enc_output, inp_padding_mask, trg_combined_mask):
        attention_weights = {}
        # trg_combined_mask = create_combined_mask(trg_input_ids)

        # 1. embedding and positional embedding
        input_emb = self.embedding(trg_input_ids)
        input_emb = self.positional_encoding(input_emb)
        # input_emb = positional_encoding(self.config, input_emb, input_emb.size(1))
        x = self.dropout(input_emb)

        # 2. decoder
        for i, dec in enumerate(self.decoder):
            x, block1, block2 = dec(x, trg_combined_mask, enc_output, inp_padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class DecoderLayer(nn.Module, ABC):
    """ 一个Decoder层 """
    def __init__(self, config):
        super().__init__()

        self.mha1 = MultiHeadAttention(config)
        self.mha2 = MultiHeadAttention(config)
        self.ffn = FeedForward(config)

        self.layer_norm1 = nn.LayerNorm(config.model_config.dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(config.model_config.dim, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(config.model_config.dim, eps=1e-6)

        self.dropout1 = nn.Dropout(config.model_config.drop_rate)
        self.dropout2 = nn.Dropout(config.model_config.drop_rate)
        self.dropout3 = nn.Dropout(config.model_config.drop_rate)

    def forward(self, trg_emb, combined_mask, enc_output, input_padding_mask):
        """
        Decoder的前向传播
        :param trg_emb: Decoder输入embedding    [batch_size, trg_seq_len, dim]
        :param combined_mask: Decoder input embedding的mask，综合padding与look ahead
        :param input_padding_mask: src输入的padding mask
        :param enc_output: Encoder的输出.          [batch_size, src_seq_len, dim]
        :return: 一层decoder layer输出
        """
        # 1. 第一层Self Attention，QKV都是trg emb自己
        att_output1, attn1 = self.mha1(trg_emb, trg_emb, trg_emb, combined_mask)
        att_output1 = self.dropout1(att_output1)
        output1 = self.layer_norm1(trg_emb + att_output1)

        # 2. 第二层Self Attention，Q为第一层的输出，KV为encoder的输出
        # 这一层是decoder layer关注encoder的输出序列
        att_output2, attn2 = self.mha2(output1, enc_output, enc_output, input_padding_mask)
        att_output2 = self.dropout2(att_output2)
        output2 = self.layer_norm2(output1 + att_output2)

        # 3. FeedForward层
        ffn_out = self.ffn(output2)
        ffn_out = self.dropout3(ffn_out)
        output3 = self.layer_norm3(output2 + ffn_out)

        return output3, attn1, attn2


class MultiHeadAttention(nn.Module, ABC):

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
        :param input_mask: Mask Tensor.             [batch_size, 1, 1, seq_len]
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

        # [batch_size, num_heads, seq_len_q, seq_len_k]
        matmul_qk = torch.matmul(Q, torch.transpose(K, -1, -2))
        scaled_attention_logits = matmul_qk / (K.size(-1) ** 0.5)
        # 带上mask，正常token位置+0，mask位置变为负极大，保证softmax后，mask处位置变成0，即不再注意
        if input_mask is not None:
            scaled_attention_logits += (input_mask * -1e9)
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # [batch_size, num_heads, seq_len_q, sub_head_dim]
        scaled_attention = torch.matmul(attention_weights, V)

        # 转回原来的维度 [batch_size, seq_len_q, num_heads, sub_head_dim]
        scaled_attention = torch.transpose(scaled_attention, 1, 2)
        concat_attention = scaled_attention.reshape([batch_size, -1, self.d_model])

        output = self.dense(concat_attention)
        return output, attention_weights


class FeedForward(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.model_config.dim, config.model_config.ff_size),
            nn.ReLU(),
            nn.Linear(config.model_config.ff_size, config.model_config.dim)
        )

    def forward(self, input_tensor):
        return self.ffn(input_tensor)


# if __name__ == '__main__':
#     import configuration
#     config = configuration.Config()
#     config.model_config.src_vocab_size = 10
#     config.model_config.trg_vocab_size = 12
#
#     src_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
#                             [2, 3, 4, 5, 6, 7, 0, 0]])
#
#     trg_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0, 0],
#                             [2, 3, 4, 5, 6, 7, 1, 2, 3]])
#
#     # 这里检测encoder、decoder与完整模型的输出
#     transformer = Transformer(config)
#     logits = transformer(src_seq, trg_seq[:, :-1])
#     print('Transformer output: \n', logits)
#     print('Transformer output shape: ', logits.shape)
#     print('-' * 50)
#
#     enc = EncoderLayer(config)
#     src_emb = F.embedding(input=src_seq, weight=torch.randn(config.model_config.src_vocab_size,
#                                                             config.model_config.dim))
#     input_padding_mask = create_padding_mask(src_seq)
#     enc_out, enc_attn = enc(src_emb, input_padding_mask)
#     print('Encoder output shape: \n', enc_out.shape)
#     print('Encoder Attention shape: ', enc_attn.shape)
#     print('Encoder Attention: \n', enc_attn)
#     print('-' * 50)
#
#     dec = DecoderLayer(config)
#     trg_emb = F.embedding(input=trg_seq, weight=torch.randn(config.model_config.trg_vocab_size,
#                                                             config.model_config.dim))
#     combined_mask = create_combined_mask(trg_seq)
#     dec_out, dec_attn1, dec_attn2 = dec(trg_emb, combined_mask, enc_out, input_padding_mask)
#     print('Decoder output shape: \n', dec_out.shape)
#     print('Decoder Attention1 shape: ', dec_attn1.shape)
#     print('Decoder Attention1: \n', dec_attn1)
#     print('Decoder Attention2 shape: ', dec_attn2.shape)
#     print('Decoder Attention2: \n', dec_attn2)

    # dec = TransformerDecoder(config)
    #
    # inp_padding_mask = create_padding_mask(src_seq)
    # combined_mask = create_combined_mask(trg_seq)
    #
    # enc_out = enc(src_seq)
    #
    # print(combined_mask)
    # print(inp_padding_mask)
    # print('---------')
    # print(enc_out)
    # print(enc_out.shape)
    # print('---------')
    # dec_out, attn = dec(trg_seq, enc_out, inp_padding_mask)
    # print(dec_out)
    # print(dec_out.shape)
    # for block_name, attn_weights in attn.items():
    #     print(f"{block_name}.shape: {attn_weights.shape}")
