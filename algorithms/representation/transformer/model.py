"""Transformer encoder-decoder (Vaswani et al., NeurIPS 2017, Figure 1).

Code -> paper mapping:
- Section 3.1: encoder/decoder stacks of N identical layers, each sub-layer
  wrapped as LayerNorm(x + Sublayer(x)) (residual connection + layer norm).
- Section 3.2.1, Eq. (1): scaled dot-product attention.
- Section 3.2.2: multi-head attention, d_k = d_v = d_model / h.
- Section 3.2.3: padding masks and the autoregressive subsequent mask.
- Section 3.3, Eq. (2): position-wise feed-forward network.
- Section 3.4: embeddings scaled by sqrt(d_model); input/output embeddings
  and the pre-softmax projection share one weight matrix.
- Section 3.5: sinusoidal positional encoding.
- Section 5.4: residual dropout on every sub-layer output and on the
  embedding + positional-encoding sums.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Section 3.2.1, Eq. (1): Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

    mask entries that are False are set to -inf before the softmax
    (Section 3.2.3), removing the corresponding positions from the weights.
    """
    d_k = query.size(-1)
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ value


def subsequent_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """Section 3.2.3: position i may only attend to positions <= i."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def make_src_mask(src: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Mask out padding keys; broadcasts to (batch, heads, query_len, key_len)."""
    return (src != pad_id)[:, None, None, :]


def make_tgt_mask(tgt: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Padding mask combined with the subsequent mask for decoder input."""
    pad_mask = (tgt != pad_id)[:, None, :, None]
    return pad_mask & subsequent_mask(tgt.size(1), tgt.device)


class MultiHeadAttention(nn.Module):
    """Section 3.2.2: MultiHead(Q,K,V) = Concat(head_1..head_h) W^O,
    head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(query.size(0), query.size(1), -1)
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    """Section 3.3, Eq. (2): FFN(x) = max(0, x W_1 + b_1) W_2 + b_2."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Section 3.1: multi-head self-attention + FFN, each with residual+norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class DecoderLayer(nn.Module):
    """Section 3.1 + 3.2.3: masked self-attention, encoder-decoder attention,
    and FFN, each with residual+norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.dropout(self.ffn(x)))


class Encoder(nn.Module):
    """Section 3.1: stack of N identical encoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Section 3.1: stack of N identical decoder layers."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    """Section 3.5: PE(pos,2i) = sin(pos / 10000^(2i/d_model)),
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))."""

    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class Transformer(nn.Module):
    """Full encoder-decoder (Figure 1): embed -> encode -> decode -> generate."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = self.src_embed  # Section 3.4: shared embedding matrix
        self.pos_enc = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_heads, d_ff, dropout)
        self.generator = nn.Linear(d_model, vocab_size)
        # Section 3.4: the pre-softmax projection shares the embedding weight.
        self.generator.weight = self.src_embed.weight
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Section 3.4: embedding weights are multiplied by sqrt(d_model).
        return self.encoder(
            self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model)), src_mask
        )

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(
            self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model)),
            memory,
            src_mask,
            tgt_mask,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced logits over the vocabulary, (batch, tgt_len, vocab)."""
        memory = self.encode(src, src_mask)
        return self.generator(self.decode(tgt, memory, src_mask, tgt_mask))
