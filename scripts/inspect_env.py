"""Play random legal actions in divine-sts2 and dump real observations for tokenizer design.

Writes data/inspect/episode_<n>.jsonl (one line per step: observation, legal actions, reward)
and prints random-bot win rate, episode length, and every observation key path seen.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sts2_native_sim.gym import DEFAULT_SCENARIO, Sts2NativeVectorEnv


def key_paths(obj, prefix: str = "") -> set[str]:
    """Flatten nested dict/list structure into key paths like 'combat.creatures[].hp'."""
    paths: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            paths.add(p)
            paths |= key_paths(v, p)
    elif isinstance(obj, list):
        for item in obj:
            paths |= key_paths(item, f"{prefix}[]")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encounter", default="first", help="encounter model_id, e.g. NIBBITS_WEAK")
    ap.add_argument("--out", type=Path, default=Path("data/inspect"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    paths: set[str] = set()
    action_types: Counter[str] = Counter()
    wins, lengths, returns = 0, [], []

    scenario = {**DEFAULT_SCENARIO, "encounter": args.encounter}
    with Sts2NativeVectorEnv(workers=1, default_scenario=scenario, seed=args.seed) as env:
        for ep in range(args.episodes):
            obs, info = env.reset()
            total, steps, victory = 0.0, 0, False
            with (args.out / f"episode_{ep}.jsonl").open("w", encoding="utf-8") as f:
                for steps in range(1, args.max_steps + 1):
                    legal = info["legal_actions"][0]
                    paths |= key_paths(obs[0])
                    action_types.update(a.get("type", a.get("kind", "?")) for a in legal)
                    action = rng.choice(info["legal_action_ids"][0])
                    f.write(json.dumps({"obs": obs[0], "legal_actions": legal, "action": action}) + "\n")
                    obs, rew, term, _, info = env.step([action])
                    total += rew[0]
                    if term[0]:
                        victory = bool(env.current_states[0].get("victory", False))
                        break
            wins += victory
            lengths.append(steps)
            returns.append(total)
            print(f"episode {ep}: {'WIN ' if victory else 'LOSS'}  steps {steps:3d}  return {total:7.1f}")

    n = args.episodes
    print(f"\nrandom bot: win rate {wins / n:.0%}  mean steps {sum(lengths) / n:.1f}  mean return {sum(returns) / n:.1f}")
    print("\nlegal action types seen:", dict(action_types))
    print("\nobservation key paths:")
    for p in sorted(paths):
        print("  " + p)


if __name__ == "__main__":
    main()
