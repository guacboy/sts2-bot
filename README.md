# sts2-bot

A small transformer (10M to 100M parameters) built and trained from scratch in PyTorch to play Slay the Spire 2. The game rules come from [divine-sts2](https://github.com/favet/divine-sts2), which runs the real game logic headless from a local Steam install.

Training runs in two stages: imitation learning first, then PPO. The initial scope is combat only, with one character in Act 1.

## Status

- [x] Warmup: GPT-style transformer and char-level training loop
- [x] divine-sts2 setup (bootstrap + `doctor`)
- [x] Inspect real observations with random legal actions
- [x] State tokenizer, action head with legal action masking
- [ ] Imitation learning
- [ ] PPO fine tuning
- [ ] Milestone: win NIBBITS_NORMAL more often than a random bot (random wins 0% of 20 games)

## Setup

Requires Windows 10/11 x64, Python 3.10+, and an NVIDIA GPU (an RTX 50 series card needs the CUDA 12.8 PyTorch build).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
git config core.hooksPath .githooks   # enables the pytest pre-push gate
```

### Game environment (divine-sts2)

divine-sts2 is cloned into `external/` (git ignored). Its workers run the game headless through Godot 4.5.1 .NET, and its scripts download .NET 9 and Godot into `external/divine-sts2/.tools/` without a system install. The scripts work in Windows PowerShell 5.1.

```powershell
git clone https://github.com/favet/divine-sts2 external/divine-sts2
$d = "external\divine-sts2"
powershell -ExecutionPolicy Bypass -File $d\scripts\install-dotnet-9.ps1
powershell -ExecutionPolicy Bypass -File $d\scripts\install-godot-4.5.1.ps1
# Godot loads the Debug assembly when running from source, so build both configurations
powershell -ExecutionPolicy Bypass -File $d\scripts\build-persistent-server.ps1 -Configuration Release
powershell -ExecutionPolicy Bypass -File $d\scripts\build-persistent-server.ps1 -Configuration Debug
pip install -e $d
cd $d; python -m sts2_native_sim.cli doctor --deep --json   # expect "ok": true
```

The game is auto detected from Steam. Set `STS2_GAME_ROOT` in `.env` for a custom library path. The upstream `bootstrap.ps1` does the same steps, but also installs ruff and mypy.

## Layout

| Path | Purpose |
| --- | --- |
| `src/sts2bot/model.py` | GPT model (pre-norm blocks, optional non-causal attention, `hidden()` for future policy/value heads) |
| `src/sts2bot/warmup.py` | Char-level training loop with TensorBoard logging |
| `src/sts2bot/encoding.py` | Observation tokenizer (`encode`), growing `Vocab`, padded batching (`collate`) |
| `src/sts2bot/policy.py` | `CombatPolicy`: non-causal transformer that scores each legal action, plus a value head |
| `src/sts2bot/evaluate.py` | Plays real combats with a random bot or a policy, reports win rate and HP kept |
| `scripts/inspect_env.py` | Plays random legal actions, dumps observations to `data/inspect/`, reports win rate |
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

## Environment findings

Run `python scripts/inspect_env.py --encounter NIBBITS_NORMAL --episodes 20` to reproduce.

- The default scenario is Ironclad with the starter deck (5 Strike, 4 Defend, Bash) and 80 HP. `encounter: "first"` resolves to the Bygone Effigy elite.
- Action IDs are strings tied to card instances (`play:strike-2:target:1`, `end_turn`), so the policy scores each legal action instead of predicting from a fixed vocabulary.
- The first decision of a combat comes before the opening draw, with only `end_turn` legal.
- The env does not auto reset finished workers, and it never truncates.
- Built-in reward: enemy HP removed + 1.5 x player HP change, +100 on victory, -50 on defeat.

Random bot win rate over 20 games (starter deck, ascension 0):

| Encounter group | Random win rate |
| --- | --- |
| Act 1 weak (Nibbits, Slimes, Toadpoles, ...) | 75% to 100% |
| Act 1 normal, most (Nibbits, Slimes, Cultists, Two Tailed Rats, ...) | 0% to 10% |
| Act 1 normal, easier (Inklets, Punch Construct, Mawler, Haunted Ship) | 30% to 60% |
| Act 1 elites | 0% |

## Combat policy

Each observation becomes a set of tokens. Every token sums four embeddings:

- **kind**: global state, player, enemy, intent, power, a card in hand/draw/discard/exhaust, relic, potion, or legal action
- **ids**: up to two categorical ids from a shared `Vocab` (card or monster `model_id`, action kind, intent type and move)
- **slot**: the creature a token refers to (0 none, 1 player, 2 and up for enemies), which links powers, intents, and attack targets to their creature
- **features**: six scaled numbers (HP, block, energy cost, intent damage, ...)

There are no positional embeddings, so token order does not affect the output. Each legal action is its own token, and the policy head scores only those tokens. Illegal moves get no logit at all, so masking happens by construction. The value head reads the global token.

Compare a random bot with an untrained (or trained) policy on real combats:

```powershell
python -m sts2bot.evaluate --encounter NIBBITS_NORMAL --episodes 20
python -m sts2bot.evaluate --checkpoint checkpoints/policy.pt
```

## Tests

```powershell
python -m pytest
```

## Notes

- divine-sts2 checks game file hashes. Disable Steam auto updates for StS2, or train from a copied game folder.
- Decompiled game code belongs to Mega Crit. Keep anything derived from it private and noncommercial.
