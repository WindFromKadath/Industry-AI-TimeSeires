"""Train + evaluate the Transformer on a registered dataset.

Run from this directory:
    uv run python train.py                              # synthetic default
    uv run python train.py --dataset wikitext-2-raw --epochs 8 --batch-size 128

Every run writes one record to
outputs/transformer/<dataset>/metrics.json so results stay comparable
across datasets and algorithms.
"""

import argparse
import json
import math
from datetime import datetime, timezone

import pandas as pd
import torch
from config import Config
from model import Transformer, make_src_mask, make_tgt_mask
from paths import OUTPUT_DIR
from plot import plot_history

from data import BOS_ID, PAD_ID, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic-copy-task")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--d-model", type=int, default=Config.d_model)
    parser.add_argument("--n-heads", type=int, default=Config.n_heads)
    parser.add_argument("--n-layers", type=int, default=Config.n_layers)
    parser.add_argument("--d-ff", type=int, default=Config.d_ff)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--warmup-steps", type=int, default=Config.warmup_steps)
    parser.add_argument("--label-smoothing", type=float, default=Config.label_smoothing)
    return parser.parse_args()


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Section 5.3, Eq. (3): lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""
    return d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)


def label_smoothed_nll(
    logits: torch.Tensor, target: torch.Tensor, eps: float, pad_id: int = PAD_ID
) -> torch.Tensor:
    """Section 5.4: cross-entropy with label smoothing eps_ls over non-PAD tokens."""
    log_probs = logits.log_softmax(dim=-1)
    nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    smooth = -log_probs.mean(dim=-1)
    mask = target != pad_id
    loss = (1.0 - eps) * nll + eps * smooth
    return (loss * mask).sum() / mask.sum()


def build_batch(
    sequences: list[list[int]],
    device: torch.device,
    objective: str = "copy",
    context_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn sequences into padded (src, decoder_input, target) tensors.

    objective="copy": tgt = src. objective="continuation": src holds the
    first `context_len` tokens, tgt the rest. decoder_input is
    [BOS] + target[:-1], so position i predicts target i (Section 3.1:
    outputs are offset by one position).
    """
    if objective == "continuation":
        pairs = [(s[:context_len], s[context_len:]) for s in sequences]
    else:
        pairs = [(s, s) for s in sequences]
    src_w = max(len(s) for s, _ in pairs)
    tgt_w = max(len(t) for _, t in pairs)
    src = torch.full((len(pairs), src_w), PAD_ID, dtype=torch.long)
    tgt_in = torch.full((len(pairs), tgt_w), PAD_ID, dtype=torch.long)
    tgt_out = torch.full((len(pairs), tgt_w), PAD_ID, dtype=torch.long)
    for row, (s, t) in enumerate(pairs):
        if not t:
            continue
        src[row, : len(s)] = torch.tensor(s)
        tgt_in[row, : len(t)] = torch.tensor([BOS_ID] + t[:-1])
        tgt_out[row, : len(t)] = torch.tensor(t)
    return src.to(device), tgt_in.to(device), tgt_out.to(device)


@torch.no_grad()
def greedy_decode(
    model: Transformer, src: torch.Tensor, max_len: int, device: torch.device
) -> torch.Tensor:
    """Autoregressive inference: start from BOS, take argmax step by step."""
    model.eval()
    src = src.to(device)
    src_mask = make_src_mask(src, PAD_ID)
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=device)
    for _ in range(max_len):
        out = model.decode(ys, memory, src_mask, make_tgt_mask(ys, PAD_ID))
        next_token = model.generator(out[:, -1]).argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_token], dim=1)
    return ys[:, 1:]


@torch.no_grad()
def evaluate(
    model: Transformer,
    sequences: list[list[int]],
    batch_size: int,
    device: torch.device,
    objective: str,
    context_len: int,
) -> dict[str, float]:
    """Teacher-forced token accuracy and perplexity; the copy objective also
    reports exact sequence accuracy under greedy decoding."""
    model.eval()
    correct_tokens = total_tokens = exact_sequences = 0
    nll_sum = 0.0
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        src, tgt_in, tgt_out = build_batch(chunk, device, objective, context_len)
        logits = model(
            src, tgt_in, make_src_mask(src, PAD_ID), make_tgt_mask(tgt_in, PAD_ID)
        )
        mask = tgt_out != PAD_ID
        correct_tokens += ((logits.argmax(dim=-1) == tgt_out) & mask).sum().item()
        total_tokens += mask.sum().item()
        nll_sum += torch.nn.functional.cross_entropy(
            logits[mask], tgt_out[mask], reduction="sum"
        ).item()
        if objective == "copy":
            decoded = greedy_decode(model, src, src.size(1), device)
            exact_sequences += (decoded == src).all(dim=1).sum().item()
    metrics = {
        "test_token_accuracy": round(correct_tokens / total_tokens, 4),
        "test_perplexity": round(math.exp(nll_sum / total_tokens), 1),
    }
    if objective == "copy":
        metrics["test_sequence_accuracy"] = round(exact_sequences / len(sequences), 4)
    return metrics


def main() -> None:
    args = parse_args()
    config = Config(
        seed=args.seed,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
    )
    torch.manual_seed(config.seed)
    dataset = load_dataset(args.dataset, seed=config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"running on {device} with seed={config.seed}")

    model = Transformer(
        dataset.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_len,
    ).to(device)
    # Section 5.3: Adam(beta1=0.9, beta2=0.98, eps=1e-9) + Noam schedule.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)

    step = 0
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(dataset.train)).tolist()
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            chunk = [dataset.train[i] for i in order[start : start + config.batch_size]]
            src, tgt_in, tgt_out = build_batch(
                chunk, device, dataset.objective, dataset.context_len
            )
            step += 1
            lr = noam_lr(step, config.d_model, config.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            logits = model(
                src, tgt_in, make_src_mask(src, PAD_ID), make_tgt_mask(tgt_in, PAD_ID)
            )
            loss = label_smoothed_nll(logits, tgt_out, config.label_smoothing)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        mean_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "train_loss": round(mean_loss, 4), "lr": lr})
        print(f"epoch {epoch + 1}/{config.epochs}  loss={mean_loss:.4f}  lr={lr:.2e}")

    metrics = {
        "final_train_loss": history[-1]["train_loss"] if history else math.nan,
        **evaluate(
            model,
            dataset.test,
            config.batch_size,
            device,
            dataset.objective,
            dataset.context_len,
        ),
    }
    record = {
        "dataset": dataset.name,
        "params": vars(args),
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = OUTPUT_DIR / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = out_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)
    plot_history(history_csv, out_dir / "curves.png")
    out_file = out_dir / "metrics.json"
    out_file.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"metrics written to {out_file}; refresh docs/benchmarks.md with:")
    print("    uv run python scripts/collect_results.py")


if __name__ == "__main__":
    main()
