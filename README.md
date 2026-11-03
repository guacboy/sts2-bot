# sts2-bot

A small transformer (10M to 100M parameters) built and trained from scratch in PyTorch to play Slay the Spire 2. The game rules come from [divine-sts2](https://github.com/favet/divine-sts2), which runs the real game logic headless from a local Steam install.

Training runs in two stages: imitation learning first, then PPO. The initial scope is combat only, with one character in Act 1.

## Status

- [x] Warmup: GPT-style transformer and char-level training loop
- [ ] divine-sts2 setup (bootstrap + `doctor`)
- [ ] Inspect real observations with random legal actions
- [ ] State tokenizer, action head with legal action masking
- [ ] Imitation learning
- [ ] PPO fine tuning
- [ ] Milestone: beat an Act 1 fight more often than a random bot

## Setup

Requires Windows 10/11 x64, Python 3.10+, and an NVIDIA GPU (an RTX 50 series card needs the CUDA 12.8 PyTorch build).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
git config core.hooksPath .githooks   # enables the pytest pre-push gate
```

## Layout

| Path | Purpose |
| --- | --- |
| `src/sts2bot/model.py` | GPT model (pre-norm blocks, optional non-causal attention, `hidden()` for future policy/value heads) |
| `src/sts2bot/warmup.py` | Char-level training loop with TensorBoard logging |
| `tests/` | Pytest suite, run by the pre-push hook |
| `.githooks/pre-push` | Blocks pushes when tests fail |

## Warmup

Train on a synthetic digit-sorting task (`3172>1237`). It reports exact-match accuracy, so you can confirm the loop actually learns:

```powershell
python -m sts2bot.warmup
```

Train on any text file instead (e.g. tiny Shakespeare):

```powershell
python -m sts2bot.warmup --data data/input.txt --block-size 256 --n-layer 6 --n-head 6 --n-embd 384 --dropout 0.2 --lr 3e-4 --max-iters 5000
```

View the training curves with `tensorboard --logdir runs`. Checkpoints go to `checkpoints/`.

## Tests

```powershell
python -m pytest
```

## Notes

- divine-sts2 checks game file hashes. Disable Steam auto updates for StS2, or train from a copied game folder.
- Decompiled game code belongs to Mega Crit. Keep anything derived from it private and noncommercial.
