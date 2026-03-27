import torch
import torch.nn as nn
import torch.utils.data as data
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Scaled dot product attention for the num_heads heads.
        Uses PyTorch's optimized scaled_dot_product_attention
        (Flash Attention) when available.

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        mask : torch.Tensor, optional
            Padding mask, by default None. 
            If not None, shape (batch_size, window_size) and of the 
            bool dtype, with True on the positions that correspond to 
            padded / masked events. 

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        # Try Flash Attention (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_mask = None
            if mask is not None:
                batch_size = Q.shape[0]
                window_size = Q.shape[2]
                attn_mask = torch.broadcast_to(mask.unsqueeze(1), size=(batch_size, window_size, window_size))
                attn_mask = attn_mask.unsqueeze(1)  # (B, 1, W, W) - broadcasts to (B, heads, W, W)
                attn_mask = attn_mask.float().masked_fill(attn_mask.bool(), float('-inf'))
            output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            return output

        # Fallback: manual implementation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, num_heads, W, W)
        if mask is not None: # (batch_size, window_size)
            batch_size = Q.shape[0]
            window_size = Q.shape[2]
            mask = torch.broadcast_to(mask.unsqueeze(1), size = (batch_size, window_size, window_size))
            attn_scores = attn_scores.masked_fill(mask = mask.unsqueeze(1), value = -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        """MHA

        Parameters
        ----------
        Q : torch.Tensor 
            Queries. Tensor of shape (batch_size, window_size, d_model)
        K : torch.Tensor 
            Keys. Tensor of shape (batch_size, window_size, d_model)
        V : torch.Tensor 
            Values. Tensor of shape (batch_size, window_size, d_model)
        mask : torch.Tensor, optional
            Boolean mask of shape (batch_size, window_size). Entries are 
            True for the embeddings that correspond to padded events. 

        Returns
        -------
        _type_
            _description_
        """
        Q = self.split_heads(self.W_q(Q)) # (batch_size, self.num_heads, window_size, self.d_k)
        K = self.split_heads(self.W_k(K)) # (batch_size, self.num_heads, window_size, self.d_k)
        V = self.split_heads(self.W_v(V)) # (batch_size, self.num_heads, window_size, self.d_k)

        # So your mask needs to be of the shape (batch_size, self.num_heads, window_size, self_d_k)
        # and it is fed into the MHA with shape (batch_size, window_size)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

# Self attention for in decoder. Seperate class because of fixed look-ahead mask. 
class MultiHeadSelfAttentionDecoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttentionDecoder, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V):
        """Scaled dot product attention for the num_heads heads.
        Uses PyTorch's optimized scaled_dot_product_attention
        (Flash Attention) when available, with is_causal=True.

        Parameters
        ----------
        Q : torch.Tensor
            Projected and split up queries, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        K : torch.Tensor
            Projected and split up keys, shape 
            (batch_size, self.num_heads, window_size, self.d_k).
        V : torch.Tensor
            Projected and split up values, shape 
            (batch_size, self.num_heads, window_size, self.d_k).

        Returns
        -------
        output : torch.Tensor
            The result of the MHA. Shape 
            (batch_size, self.num_heads, window_size, self.d_k)
        """
        # Try Flash Attention with causal mask (PyTorch 2.0+)
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            output = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, attn_mask=None, dropout_p=0.0, is_causal=True
            )
            return output

        # Fallback: manual implementation
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, num_heads, W, W)

        window_size = Q.shape[2]
        look_ahead = torch.triu(torch.ones(1, 1, window_size, window_size), diagonal=1).bool()
        look_ahead = look_ahead.to(device)

        attn_scores = attn_scores.masked_fill(mask = look_ahead, value = -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # x : shape (batch_size, window_size, d_model)
        batch_size, seq_length, d_model = x.size()
        # x.view(...) further subdivides the innermost dim (of size d_model) into 
        # num_heads vectors of size d_k. 
        # x.view(...).transpose(1,2) transposes axis 1 and axis 2. 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) 
        # shape (batch_size, self.num_heads, window_size, self.d_k)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V):
        """MHA

        Parameters
        ----------
        Q : torch.Tensor 
            Queries. Tensor of shape (batch_size, window_size, d_model)
        K : torch.Tensor 
            Keys. Tensor of shape (batch_size, window_size, d_model)
        V : torch.Tensor 
            Values. Tensor of shape (batch_size, window_size, d_model)

        Returns
        -------
        _type_
            _description_
        """
        Q = self.split_heads(self.W_q(Q)) # (batch_size, self.num_heads, window_size, self.d_k)
        K = self.split_heads(self.W_k(K)) # (batch_size, self.num_heads, window_size, self.d_k)
        V = self.split_heads(self.W_v(V)) # (batch_size, self.num_heads, window_size, self.d_k)

        # So your mask needs to be of the shape (batch_size, self.num_heads, window_size, self_d_k)
        # and it is fed into the MHA with shape (batch_size, window_size)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x))) # (batch_size, window_size, d_model)


# ============================================================================
# KV-Caching versions for fast autoregressive inference
# ============================================================================

class MultiHeadSelfAttentionDecoderCached(nn.Module):
    """Self-attention with KV-caching for efficient autoregressive decoding.

    During inference, instead of recomputing attention over all previous
    tokens, we cache the K and V projections and only compute the new
    token's Q/K/V. No causal mask is needed because Q is always just the
    current token — it can only attend to past tokens already in the cache.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttentionDecoderCached, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x, cache_k=None, cache_v=None):
        """Forward pass with KV caching.

        Parameters
        ----------
        x : torch.Tensor
            New token(s). Shape (batch_size, seq_len, d_model).
            During cached inference seq_len = 1.
        cache_k : torch.Tensor or None
            Cached keys from prior steps.
            Shape (batch_size, num_heads, cached_len, d_k).
        cache_v : torch.Tensor or None
            Cached values from prior steps.

        Returns
        -------
        output : torch.Tensor
            Shape (batch_size, seq_len, d_model).
        new_cache_k : torch.Tensor
            Updated key cache including current step.
        new_cache_v : torch.Tensor
            Updated value cache including current step.
        """
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        # Append to cache
        if cache_k is not None and cache_v is not None:
            K = torch.cat([cache_k, K], dim=2)
            V = torch.cat([cache_v, V], dim=2)

        new_cache_k = K
        new_cache_v = V

        # Standard attention (no mask needed — Q is current token only)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        output = self.W_o(self.combine_heads(attn_output))
        return output, new_cache_k, new_cache_v


class MultiHeadCrossAttentionCached(nn.Module):
    """Cross-attention with caching for encoder K/V projections.

    The encoder K and V are projected once on the first decoding step
    and reused for all subsequent steps — avoiding redundant projection
    of the (unchanging) encoder output at every step.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttentionCached, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x, enc_output, src_mask=None, enc_cache_k=None, enc_cache_v=None):
        """Forward pass with optional encoder KV caching.

        Parameters
        ----------
        x : torch.Tensor
            Decoder query. Shape (batch_size, seq_len, d_model).
        enc_output : torch.Tensor
            Encoder output. Shape (batch_size, enc_seq_len, d_model).
        src_mask : torch.Tensor or None
            Source padding mask. Shape (batch_size, enc_seq_len).
            True for padded positions.
        enc_cache_k : torch.Tensor or None
            Pre-computed encoder keys. If provided, skips re-projection.
        enc_cache_v : torch.Tensor or None
            Pre-computed encoder values.

        Returns
        -------
        output : torch.Tensor
            Shape (batch_size, seq_len, d_model).
        K : torch.Tensor
            Encoder key cache (compute once, reuse).
        V : torch.Tensor
            Encoder value cache.
        """
        Q = self.split_heads(self.W_q(x))

        if enc_cache_k is not None and enc_cache_v is not None:
            K = enc_cache_k
            V = enc_cache_v
        else:
            K = self.split_heads(self.W_k(enc_output))
            V = self.split_heads(self.W_v(enc_output))

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if src_mask is not None:
            # (B, 1, 1, enc_len) broadcasts to (B, heads, q_len, enc_len)
            mask_expanded = src_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, V)

        output = self.W_o(self.combine_heads(attn_output))
        return output, K, V