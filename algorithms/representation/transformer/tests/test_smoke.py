"""Smoke tests: forward shapes, autoregressive masking, quick overfit."""

import torch
from config import Config
from model import Transformer, make_src_mask, make_tgt_mask
from train import build_batch

from data import BOS_ID


def tiny_model(vocab_size: int = 12) -> Transformer:
    return Transformer(
        vocab_size, d_model=32, n_heads=2, n_layers=1, d_ff=64, dropout=0.0, max_len=16
    )


def test_config_defaults():
    assert Config().seed == 42


def test_build_batch_continuation():
    """continuation: src = 前 context_len 个 token,decoder 续写其余部分."""
    src, tgt_in, tgt_out = build_batch(
        [[5, 6, 7, 8, 9, 10]], torch.device("cpu"), "continuation", context_len=3
    )
    assert src.tolist() == [[5, 6, 7]]
    assert tgt_in.tolist() == [[BOS_ID, 8, 9]]
    assert tgt_out.tolist() == [[8, 9, 10]]


def test_forward_shape():
    torch.manual_seed(0)
    model = tiny_model()
    src = torch.randint(2, 12, (2, 5))
    tgt = torch.randint(2, 12, (2, 5))
    logits = model(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
    assert logits.shape == (2, 5, 12)


def test_decoder_is_autoregressive():
    """Changing future decoder-input tokens must not change earlier logits
    (Section 3.2.3: predictions for position i depend only on positions < i)."""
    torch.manual_seed(0)
    model = tiny_model().eval()
    src = torch.randint(2, 12, (1, 6))
    tgt_a = torch.randint(2, 12, (1, 6))
    tgt_b = tgt_a.clone()
    tgt_b[0, 4:] = 2  # alter only the last two positions
    with torch.no_grad():
        logits_a = model(src, tgt_a, make_src_mask(src), make_tgt_mask(tgt_a))
        logits_b = model(src, tgt_b, make_src_mask(src), make_tgt_mask(tgt_b))
    assert torch.allclose(logits_a[0, :4], logits_b[0, :4])


def test_overfits_tiny_batch():
    """A few optimization steps on one fixed copy batch should reduce loss."""
    torch.manual_seed(0)
    model = tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    src = torch.randint(2, 12, (8, 6))
    tgt_in = torch.cat([torch.ones(8, 1, dtype=torch.long), src[:, :-1]], dim=1)
    losses = []
    for _ in range(60):
        logits = model(src, tgt_in, make_src_mask(src), make_tgt_mask(tgt_in))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), src.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
