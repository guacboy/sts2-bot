"""Play full combats in divine-sts2 with any action chooser and report win rate and HP kept."""

import argparse
import random
from collections.abc import Callable
from pathlib import Path

import torch

from sts2bot.encoding import Vocab
from sts2bot.policy import CombatPolicy, PolicyConfig

# (observation, legal_actions) -> action_id
Chooser = Callable[[dict, list[dict]], str]


def random_chooser(seed: int = 0) -> Chooser:
    rng = random.Random(seed)
    return lambda obs, legal: rng.choice(legal)["action_id"]


def policy_chooser(model: CombatPolicy, vocab: Vocab, greedy: bool = True) -> Chooser:
    return lambda obs, legal: model.act([obs], [legal], vocab, greedy=greedy)[0]


def player_hp(obs: dict) -> int:
    creatures = (obs.get("combat") or {}).get("creatures", [])
    return next((c.get("hp", 0) for c in creatures if c.get("side") == "Player"), 0)


def evaluate(choose: Chooser, encounter: str, episodes: int, seed: int = 0, max_steps: int = 500) -> dict:
    from sts2_native_sim.gym import DEFAULT_SCENARIO, Sts2NativeVectorEnv

    wins, hp_kept, steps_total = 0, [], 0
    scenario = {**DEFAULT_SCENARIO, "encounter": encounter}
    with Sts2NativeVectorEnv(workers=1, default_scenario=scenario, seed=seed) as env:
        for _ in range(episodes):
            obs, info = env.reset()
            for _ in range(max_steps):
                action = choose(obs[0], info["legal_actions"][0])
                obs, _, term, _, info = env.step([action])
                steps_total += 1
                if term[0]:
                    break
            if env.current_states[0].get("victory"):
                wins += 1
                hp_kept.append(player_hp(obs[0]))
    return {
        "encounter": encounter,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "mean_hp_on_win": sum(hp_kept) / len(hp_kept) if hp_kept else 0.0,
        "mean_steps": steps_total / episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encounter", default="NIBBITS_NORMAL")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path, help="policy checkpoint; omit to also run an untrained policy")
    args = ap.parse_args()

    results = [evaluate(random_chooser(args.seed), args.encounter, args.episodes, args.seed) | {"agent": "random"}]
    if args.checkpoint:
        ck = torch.load(args.checkpoint, weights_only=False)
        model = CombatPolicy(ck["config"])
        model.load_state_dict(ck["model"])
        vocab = Vocab(ck["vocab"]["max_size"], ck["vocab"]["itos"])
        vocab.frozen = True
        agent = "policy"
    else:
        torch.manual_seed(args.seed)
        model, vocab, agent = CombatPolicy(PolicyConfig()), Vocab(), "untrained"
    chooser = policy_chooser(model.eval(), vocab)
    results.append(evaluate(chooser, args.encounter, args.episodes, args.seed) | {"agent": agent})

    for r in results:
        print(f"{r['agent']:10s} {r['encounter']}: win rate {r['win_rate']:.0%}  "
              f"hp on win {r['mean_hp_on_win']:.1f}  steps {r['mean_steps']:.1f}")


if __name__ == "__main__":
    main()
