"""Turn divine-sts2 combat observations into token sets for the policy.

Each token is the sum of four parts, embedded by the model:
  kind   what the token is (enemy, hand card, legal action, ...)
  ident  up to two categorical ids (card/monster/power model_id, action kind, intent type)
  slot   which creature it refers to (0 none, 1 player, 2.. enemies), linking powers,
         intents and action targets to their creature
  feats  N_FEATS scaled numbers (hp, cost, damage, ...)

There are no positional embeddings: the observation is a set. Legal actions are tokens too,
in the same order as the env's legal action list, so the policy scores exactly those.
"""

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import torch

N_FEATS = 6
MAX_SLOTS = 8


class Kind(IntEnum):
    PAD = 0
    GLOBAL = 1
    PLAYER = 2
    ENEMY = 3
    INTENT = 4
    POWER = 5
    HAND = 6
    DRAW = 7
    DISCARD = 8
    EXHAUST = 9
    RELIC = 10
    POTION = 11
    ACTION = 12


PILE_KINDS = {"Hand": Kind.HAND, "Draw": Kind.DRAW, "Discard": Kind.DISCARD, "Exhaust": Kind.EXHAUST}


class Vocab:
    """String -> id table that grows until max_size, then maps unseen strings to UNK."""

    PAD, UNK = 0, 1

    def __init__(self, max_size: int = 4096, itos: list[str] | None = None):
        self.max_size = max_size
        self.itos = itos or ["<pad>", "<unk>"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.frozen = False

    def __len__(self) -> int:
        return len(self.itos)

    def __call__(self, s: str | None) -> int:
        if s is None:
            return self.PAD
        if s not in self.stoi:
            if self.frozen or len(self.itos) >= self.max_size:
                return self.UNK
            self.stoi[s] = len(self.itos)
            self.itos.append(s)
        return self.stoi[s]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"max_size": self.max_size, "itos": self.itos}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["max_size"], d["itos"])


@dataclass
class Encoded:
    kind: list[int] = field(default_factory=list)
    ident: list[int] = field(default_factory=list)
    ident2: list[int] = field(default_factory=list)
    slot: list[int] = field(default_factory=list)
    feats: list[list[float]] = field(default_factory=list)
    action_pos: list[int] = field(default_factory=list)  # token index of each legal action

    def add(self, kind: Kind, ident: int, ident2: int = Vocab.PAD, slot: int = 0, feats=()) -> int:
        f = [float(x) for x in feats][:N_FEATS]
        self.kind.append(int(kind))
        self.ident.append(ident)
        self.ident2.append(ident2)
        self.slot.append(min(slot, MAX_SLOTS - 1))
        self.feats.append(f + [0.0] * (N_FEATS - len(f)))
        return len(self.kind) - 1


def _model_id(x) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    return x.get("model_id") or x.get("id")


def _card_feats(card: dict) -> list[float]:
    return [
        card.get("energy_cost", 0) / 5,
        float(bool(card.get("costs_x"))),
        card.get("upgrades", 0),
        float(bool(card.get("cost_modified"))),
    ]


def encode(obs: dict, legal_actions: list[dict], vocab: Vocab) -> Encoded:
    enc = Encoded()
    combat = obs.get("combat") or {}
    piles = {p.get("type"): p.get("cards") or p.get("cards_unordered") or [] for p in combat.get("piles", [])}

    enc.add(Kind.GLOBAL, vocab(obs.get("decision", {}).get("kind")), feats=[
        combat.get("energy", 0) / 5,
        combat.get("max_energy", 0) / 5,
        combat.get("turn", 0) / 20,
        len(piles.get("Draw", [])) / 20,
        len(piles.get("Discard", [])) / 20,
        len(piles.get("Exhaust", [])) / 20,
    ])

    slots: dict[int, int] = {}
    enemy_slot = 2
    for cr in combat.get("creatures", []):
        is_player = cr.get("side") == "Player"
        slot = 1 if is_player else enemy_slot
        enemy_slot += not is_player
        slots[cr.get("combat_id")] = slot
        hp, max_hp = cr.get("hp", 0), max(cr.get("max_hp", 1), 1)
        enc.add(Kind.PLAYER if is_player else Kind.ENEMY, vocab(cr.get("model_id")), slot=slot,
                feats=[hp / 100, max_hp / 100, hp / max_hp, cr.get("block", 0) / 50, float(cr.get("alive", True))])
        for p in cr.get("powers", []):
            enc.add(Kind.POWER, vocab(p.get("model_id")), slot=slot, feats=[p.get("amount", 0) / 10])
        move = cr.get("next_move") or {}
        for it in move.get("intents", []):
            dmg, reps = it.get("damage") or 0, it.get("repeats") or 1
            enc.add(Kind.INTENT, vocab(it.get("intent_type")), vocab(move.get("id")), slot=slot,
                    feats=[dmg / 50, reps / 5, dmg * reps / 50])

    hand = {}
    for pile_type, kind in PILE_KINDS.items():
        for card in piles.get(pile_type, []):
            enc.add(kind, vocab(card.get("model_id")), feats=_card_feats(card))
            if kind is Kind.HAND:
                hand[card.get("instance_id")] = card

    inventory = obs.get("inventory") or {}
    for relic in inventory.get("relics") or []:
        enc.add(Kind.RELIC, vocab(_model_id(relic)))
    for potion in inventory.get("potions") or []:
        if potion is not None:
            enc.add(Kind.POTION, vocab(_model_id(potion)))

    for action in legal_actions:
        params = action.get("parameters") or {}
        card = hand.get(params.get("instance_id"), {})
        enc.action_pos.append(enc.add(
            Kind.ACTION, vocab(action.get("kind")), vocab(card.get("model_id")),
            slot=slots.get(params.get("target_id"), 0), feats=_card_feats(card) if card else (),
        ))
    return enc


@dataclass
class Batch:
    kind: torch.Tensor        # (B, T) long
    ident: torch.Tensor       # (B, T) long
    ident2: torch.Tensor      # (B, T) long
    slot: torch.Tensor        # (B, T) long
    feats: torch.Tensor       # (B, T, N_FEATS) float
    pad_mask: torch.Tensor    # (B, T) bool, True for real tokens
    action_pos: torch.Tensor  # (B, A) long, token index of each legal action (0 where padded)
    action_mask: torch.Tensor  # (B, A) bool, True for real legal actions

    def to(self, device) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in vars(self).items()})


def collate(items: list[Encoded]) -> Batch:
    B = len(items)
    T = max(len(e.kind) for e in items)
    A = max(max(len(e.action_pos) for e in items), 1)
    out = {name: torch.zeros(B, T, dtype=torch.long) for name in ("kind", "ident", "ident2", "slot")}
    feats = torch.zeros(B, T, N_FEATS)
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    action_pos = torch.zeros(B, A, dtype=torch.long)
    action_mask = torch.zeros(B, A, dtype=torch.bool)
    for b, e in enumerate(items):
        n, a = len(e.kind), len(e.action_pos)
        for name in out:
            out[name][b, :n] = torch.tensor(getattr(e, name))
        feats[b, :n] = torch.tensor(e.feats)
        pad_mask[b, :n] = True
        action_pos[b, :a] = torch.tensor(e.action_pos, dtype=torch.long)
        action_mask[b, :a] = True
    return Batch(**out, feats=feats, pad_mask=pad_mask, action_pos=action_pos, action_mask=action_mask)
