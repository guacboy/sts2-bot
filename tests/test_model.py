import torch

from sts2bot.model import GPT, GPTConfig
from sts2bot.warmup import CharTokenizer, main, sort_corpus


def tiny_cfg(**kw) -> GPTConfig:
    return GPTConfig(**{"vocab_size": 13, "block_size": 16, "n_layer": 2, "n_head": 2, "n_embd": 32, **kw})


def test_forward_shapes_and_loss():
    model = GPT(tiny_cfg())
    x = torch.randint(0, 13, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 13)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_causal_mask_blocks_future_tokens():
    torch.manual_seed(0)
    model = GPT(tiny_cfg()).eval()
    x = torch.randint(0, 13, (1, 8))
    y = x.clone()
    y[0, -1] = (x[0, -1] + 1) % 13
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)


def test_non_causal_sees_future_tokens():
    torch.manual_seed(0)
    model = GPT(tiny_cfg(causal=False)).eval()
    x = torch.randint(0, 13, (1, 8))
    y = x.clone()
    y[0, -1] = (x[0, -1] + 1) % 13
    a, _ = model(x)
    b, _ = model(y)
    assert not torch.allclose(a[:, 0], b[:, 0])


def test_overfits_single_batch():
    torch.manual_seed(0)
    model = GPT(tiny_cfg())
    x = torch.randint(0, 13, (4, 16))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


def test_generate_appends_tokens():
    model = GPT(tiny_cfg())
    out = model.generate(torch.zeros(2, 3, dtype=torch.long), 20)
    assert out.shape == (2, 23)


def test_char_tokenizer_roundtrip():
    text = sort_corpus(10, 4, seed=0)
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == 12  # 10 digits, '>' and '\n'


def test_warmup_smoke(tmp_path):
    metrics = main([
        "--device", "cpu", "--max-iters", "20", "--eval-interval", "10", "--eval-iters", "2",
        "--n-layer", "1", "--n-embd", "32", "--block-size", "32", "--batch-size", "8",
        "--logdir", str(tmp_path / "runs"), "--out", str(tmp_path / "ck.pt"),
    ])
    assert {"train", "val", "sort_acc"} <= metrics.keys()
    assert (tmp_path / "ck.pt").exists()
