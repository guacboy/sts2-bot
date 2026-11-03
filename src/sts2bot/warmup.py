"""Warmup: train the GPT as a character-level language model.

With --data, trains on any text file (e.g. tiny Shakespeare).
Without it, trains on a synthetic digit-sorting corpus ("3172>1237") and reports
exact-match sorting accuracy, a quick signal that the training loop works.
"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from sts2bot.model import GPT, GPTConfig


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def sort_example(rng: random.Random, n_digits: int) -> str:
    digits = "".join(rng.choice("0123456789") for _ in range(n_digits))
    return f"{digits}>{''.join(sorted(digits))}\n"


def sort_corpus(n_examples: int, n_digits: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(sort_example(rng, n_digits) for _ in range(n_examples))


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], args, device: str) -> dict[str, float]:
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.empty(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sort_accuracy(model: GPT, tok: CharTokenizer, n_digits: int, n: int, device: str) -> float:
    model.eval()
    rng = random.Random(12345)
    correct = 0
    for _ in range(n):
        ex = sort_example(rng, n_digits)
        prompt = ex[: n_digits + 1]  # "3172>"
        # Prefix with a newline so the prompt looks like the start of a corpus line.
        idx = torch.tensor([tok.encode("\n" + prompt)], device=device)
        out = tok.decode(model.generate(idx, n_digits)[0].tolist())
        correct += out[-n_digits:] == ex[n_digits + 1: 2 * n_digits + 1]
    model.train()
    return correct / n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, help="text file to train on; omit for the synthetic sort task")
    p.add_argument("--n-digits", type=int, default=6)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--logdir", type=Path, default=Path("runs/warmup"))
    p.add_argument("--out", type=Path, default=Path("checkpoints/warmup.pt"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    text = args.data.read_text(encoding="utf-8") if args.data else sort_corpus(50_000, args.n_digits, args.seed)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    splits = {"train": data[:split], "val": data[split:]}

    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(cfg).to(args.device)
    print(f"params: {model.num_params() / 1e6:.2f}M  device: {args.device}  vocab: {tok.vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_iters)
    writer = SummaryWriter(args.logdir)
    metrics: dict[str, float] = {}
    t0 = time.time()

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            metrics = estimate_loss(model, splits, args, args.device)
            msg = f"iter {it:5d}  train {metrics['train']:.4f}  val {metrics['val']:.4f}"
            if not args.data:
                metrics["sort_acc"] = sort_accuracy(model, tok, args.n_digits, 200, args.device)
                msg += f"  sort_acc {metrics['sort_acc']:.2%}"
            for k, v in metrics.items():
                writer.add_scalar(f"warmup/{k}", v, it)
            print(f"{msg}  ({time.time() - t0:.1f}s)")
        if it == args.max_iters:
            break
        x, y = get_batch(splits["train"], args.block_size, args.batch_size, args.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    writer.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "chars": tok.chars}, args.out)
    if args.data:
        idx = torch.tensor([tok.encode(text[:1])], device=args.device)
        print("--- sample ---\n" + tok.decode(model.generate(idx, 300, greedy=False)[0].tolist()))
    return metrics


if __name__ == "__main__":
    main()
