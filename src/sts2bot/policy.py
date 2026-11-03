"""Combat policy: non-causal transformer over the observation token set, scoring each legal action."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from sts2bot.encoding import MAX_SLOTS, N_FEATS, Batch, Encoded, Kind, Vocab, collate, encode
from sts2bot.model import Block, GPTConfig


@dataclass
class PolicyConfig:
    vocab_size: int = 4096
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CombatPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        self.kind_emb = nn.Embedding(len(Kind), d)
        self.id_emb = nn.Embedding(cfg.vocab_size, d, padding_idx=Vocab.PAD)
        self.slot_emb = nn.Embedding(MAX_SLOTS, d)
        self.feat_proj = nn.Linear(N_FEATS, d)
        block_cfg = GPTConfig(vocab_size=cfg.vocab_size, n_layer=cfg.n_layer, n_head=cfg.n_head,
                              n_embd=d, dropout=cfg.dropout, causal=False)
        self.blocks = nn.ModuleList(Block(block_cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.policy_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits (B, A) with -inf on padded actions, value (B,))."""
        x = (self.kind_emb(batch.kind) + self.id_emb(batch.ident) + self.id_emb(batch.ident2)
             + self.slot_emb(batch.slot) + self.feat_proj(batch.feats))
        for block in self.blocks:
            x = block(x, batch.pad_mask)
        h = self.ln_f(x)
        action_h = h.gather(1, batch.action_pos.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        logits = self.policy_head(action_h).squeeze(-1).masked_fill(~batch.action_mask, float("-inf"))
        value = self.value_head(h[:, 0]).squeeze(-1)  # GLOBAL token is always first
        return logits, value

    @torch.no_grad()
    def act(
        self, observations: list[dict], legal_actions: list[list[dict]], vocab: Vocab, greedy: bool = False
    ) -> list[str]:
        """Pick one legal action_id per observation."""
        items: list[Encoded] = [encode(o, la, vocab) for o, la in zip(observations, legal_actions)]
        device = next(self.parameters()).device
        logits, _ = self(collate(items).to(device))
        idx = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        return [la[i]["action_id"] for la, i in zip(legal_actions, idx.tolist())]
