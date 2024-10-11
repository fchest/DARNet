import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # print("seq_len",seq_len,self.num_heads)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale

        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.embed_layer = nn.Sequential(nn.Conv2d(1, d_model * 4, kernel_size=(1, 8), padding='same'),
                                         nn.BatchNorm2d(d_model * 4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(d_model * 4, d_model, kernel_size=(c_in, 1), padding='valid'),
            nn.BatchNorm2d(d_model),
            nn.GELU())
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.embed_layer(x)
        x = self.embed_layer2(x).squeeze(2)
        x = x.permute(0, 2, 1)
        x = x + self.position_embedding(x)
        return x


class Refine(nn.Module):
    def __init__(self, c_in):
        super(Refine, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class MyAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention_layer = Attention(emb_size, num_heads, dropout=0.1)

    def forward(self, x):
        x = self.attention_layer(x)
        return x


class AttnRefine(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.attention = MyAttention(emb_size, num_heads)
        self.conv_layer = Refine(emb_size)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(emb_size, 4)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x_src = self.attention(x)
        x_src = self.conv_layer(x_src)
        gap = self.gap(x_src.permute(0, 2, 1))
        out = self.out(self.flatten(gap))
        # print(out.shape)
        return x_src, out


class DARNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        d_model = 16
        emb_size = d_model
        num_heads = 8

        self.token_embedding = TokenEmbedding(c_in=channel_size, d_model=d_model)

        self.flatten = nn.Flatten()
        self.out = nn.Linear(8, 2)
        self.stack1 = AttnRefine(emb_size, num_heads)
        self.stack2 = AttnRefine(emb_size, num_heads)

    def forward(self, x):
        x_src = self.token_embedding(x)

        new_x = []
        x_src1, new_src1 = self.stack1(x_src)
        new_x.append(new_src1)

        x_src2, new_src2 = self.stack2(x_src1)
        new_x.append(new_src2)

        out = torch.cat(new_x, -1)
        out = self.flatten(out)
        out = self.out(out)
        return out