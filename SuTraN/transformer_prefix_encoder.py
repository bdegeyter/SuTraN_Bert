import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from SuTraN.layers import MultiHeadAttention, PositionWiseFeedForward



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pre_ln=False):
        super(EncoderLayer, self).__init__()
        self.pre_ln = pre_ln
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        if self.pre_ln:
            x2 = self.norm1(x)
            x = x + self.dropout(self.self_attn(x2, x2, x2, mask))
            x2 = self.norm2(x)
            x = x + self.dropout(self.feed_forward(x2))
        else:
            attn_output = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
        return x