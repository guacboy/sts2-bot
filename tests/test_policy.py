import json
import random
from pathlib import Path

import torch

from sts2bot.encoding import Kind, Vocab, collate, encode
from sts2bot.policy import CombatPolicy, PolicyConfig

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "nibbits_obs.json").read_text())
OBS, LEGAL = FIXTURE["obs"], FIXTURE["legal_actions"]


def small_policy() -> CombatPolicy:
    torch.manual_seed(0)
    return CombatPolicy(PolicyConfig(vocab_size=64, n_layer=2, n_head=2, n_embd=32)).eval()


def test_encode_token_counts():
    enc = encode(OBS, LEGAL, Vocab())
    counts = {k: enc.kind.count(int(k)) for k in Kind}
    assert counts[Kind.GLOBAL] == 1 and enc.kind[0] == Kind.GLOBAL
    assert counts[Kind.PLAYER] == 1 and counts[Kind.ENEMY] == 2
    assert counts[Kind.INTENT] == 2 and counts[Kind.POWER] == 1
    assert counts[Kind.HAND] == 3 and counts[Kind.DRAW] == 5 and counts[Kind.DISCARD] == 2
    assert counts[Kind.ACTION] == len(LEGAL) == 5
    assert [enc.kind[p] for p in enc.action_pos] == [Kind.ACTION] * 5


def test_action_targets_map_to_enemy_slots():
    enc = encode(OBS, LEGAL, Vocab())
    slots = [enc.slot[p] for p in enc.action_pos]
    assert slots == [2, 3, 0, 0, 0]  # strike->enemy 1, strike->enemy 2, defend, defend, end_turn


def test_frozen_vocab_maps_unknown_to_unk():
    vocab = Vocab()
    vocab("STRIKE_IRONCLAD")
    vocab.frozen = True
    assert vocab("NEVER_SEEN") == Vocab.UNK
    assert vocab("STRIKE_IRONCLAD") == 2


def test_vocab_roundtrip(tmp_path):
    vocab = Vocab()
    encode(OBS, LEGAL, vocab)
    vocab.save(tmp_path / "v.json")
    loaded = Vocab.load(tmp_path / "v.json")
    assert loaded.itos == vocab.itos


def test_logits_only_on_legal_actions():
    vocab = Vocab()
    short = LEGAL[-1:]  # only end_turn
    batch = collate([encode(OBS, LEGAL, vocab), encode(OBS, short, vocab)])
    logits, value = small_policy()(batch)
    assert logits.shape == (2, 5) and value.shape == (2,)
    assert torch.isfinite(logits[0]).all()
    assert torch.isfinite(logits[1, 0]) and torch.isinf(logits[1, 1:]).all()
    probs = logits.softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(2))


def test_padding_does_not_change_outputs():
    vocab = Vocab()
    model = small_policy()
    alone = collate([encode(OBS, LEGAL, vocab)])
    bigger_obs = json.loads(json.dumps(OBS))
    bigger_obs["combat"]["piles"][2]["cards"] *= 4  # longer discard pile -> longer sequence
    padded = collate([encode(OBS, LEGAL, vocab), encode(bigger_obs, LEGAL, vocab)])
    a_logits, a_value = model(alone)
    p_logits, p_value = model(padded)
    assert torch.allclose(a_logits[0], p_logits[0], atol=1e-5)
    assert torch.allclose(a_value[0], p_value[0], atol=1e-5)


def test_token_order_does_not_matter():
    vocab = Vocab()
    model = small_policy()
    enc = encode(OBS, LEGAL, vocab)
    perm = list(range(1, len(enc.kind)))  # keep GLOBAL first, shuffle the rest
    random.Random(0).shuffle(perm)
    perm = [0] + perm
    shuffled = encode(OBS, LEGAL, vocab)
    for name in ("kind", "ident", "ident2", "slot", "feats"):
        setattr(shuffled, name, [getattr(enc, name)[i] for i in perm])
    shuffled.action_pos = [perm.index(p) for p in enc.action_pos]
    a, _ = model(collate([enc]))
    b, _ = model(collate([shuffled]))
    assert torch.allclose(a, b, atol=1e-5)


def test_learns_to_prefer_an_action():
    vocab = Vocab()
    model = small_policy().train()
    batch = collate([encode(OBS, LEGAL, vocab)])
    target = torch.tensor([1])  # strike the second Nibbit
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(50):
        logits, _ = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert model.eval().act([OBS], [LEGAL], vocab, greedy=True) == [LEGAL[1]["action_id"]]
