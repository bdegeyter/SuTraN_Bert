import torch
import torch.nn as nn
from SuTraN.layers import (
    MultiHeadAttention, PositionWiseFeedForward, MultiHeadSelfAttentionDecoder,
    MultiHeadSelfAttentionDecoderCached, MultiHeadCrossAttentionCached,
)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, activation="relu"):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadSelfAttentionDecoder(d_model, num_heads)

        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class DecoderLayerCached(nn.Module):
    """Decoder layer with KV-caching for efficient autoregressive inference.

    Maintains caches for both self-attention (growing decoder history)
    and cross-attention (static encoder projections) to avoid redundant
    computation at each decoding step.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout, activation="relu"):
        super(DecoderLayerCached, self).__init__()

        self.self_attn = MultiHeadSelfAttentionDecoderCached(d_model, num_heads)
        self.cross_attn = MultiHeadCrossAttentionCached(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask,
                self_attn_cache_k=None, self_attn_cache_v=None,
                cross_attn_cache_k=None, cross_attn_cache_v=None):
        """Forward pass with KV caching.

        Parameters
        ----------
        x : torch.Tensor
            New token embedding. Shape (batch_size, 1, d_model).
        enc_output : torch.Tensor
            Encoder output. Shape (batch_size, enc_seq_len, d_model).
        src_mask : torch.Tensor
            Source padding mask. Shape (batch_size, enc_seq_len).
        self_attn_cache_k/v : torch.Tensor or None
            Cached self-attention K/V from prior steps.
        cross_attn_cache_k/v : torch.Tensor or None
            Cached cross-attention K/V (encoder projections).

        Returns
        -------
        x : torch.Tensor
            Layer output. Shape (batch_size, 1, d_model).
        new_self_k, new_self_v : torch.Tensor
            Updated self-attention caches.
        cross_k, cross_v : torch.Tensor
            Cross-attention caches (unchanged after first step).
        """
        # Self-attention with cache
        attn_output, new_self_k, new_self_v = self.self_attn(
            x, cache_k=self_attn_cache_k, cache_v=self_attn_cache_v
        )
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-attention with encoder cache
        attn_output, cross_k, cross_v = self.cross_attn(
            x, enc_output, src_mask,
            enc_cache_k=cross_attn_cache_k, enc_cache_v=cross_attn_cache_v
        )
        x = self.norm2(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, new_self_k, new_self_v, cross_k, cross_v