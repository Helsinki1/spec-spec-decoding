"""Step 1: ordinary speculative decoding, including exact stochastic sampling.

Models take token IDs [batch, time] and return logits [batch, time, vocabulary].
No KV cache: every call recomputes its prefix to keep the algorithm visible.
"""

from dataclasses import dataclass

import torch


@dataclass
class Draft:
    tokens: list[int]
    probs: torch.Tensor  # [lookahead, vocabulary], on CPU


@torch.inference_mode()
def probabilities(model, tokens, temperature):
    device = next(model.parameters()).device
    logits = model(torch.tensor([tokens], device=device))[0].double().cpu()
    if temperature == 0:
        return torch.zeros_like(logits).scatter_(-1, logits.argmax(-1, keepdim=True), 1)
    return (logits / temperature).softmax(-1)


def sample(probs, rng):
    return torch.multinomial(probs, 1, generator=rng).item()


def make_draft(model, prefix, lookahead, temperature, rng):
    tokens, probs = [], []
    for _ in range(lookahead):
        q = probabilities(model, prefix + tokens, temperature)[-1]
        tokens.append(sample(q, rng))
        probs.append(q)
    return Draft(tokens, torch.stack(probs))


def verify(target, prefix, proposal, temperature, rng):
    # Logits BEFORE each proposed token predict that token. The final row
    # predicts the bonus token if every proposal is accepted.
    p = probabilities(target, prefix + proposal.tokens, temperature)[len(prefix) - 1 :]
    for k, token in enumerate(proposal.tokens):
        q = proposal.probs[k]
        accept = min(1.0, (p[k, token] / q[token]).item())
        if torch.rand((), generator=rng).item() >= accept:
            residual = (p[k] - q).clamp_min(0)
            return k, sample(residual / residual.sum(), rng)
    return len(proposal.tokens), sample(p[-1], rng)


def append_tokens(tokens, committed, limit, eos_token_id):
    """Commit up to the output budget or first generated EOS; report completion."""
    committed = committed[: limit - len(tokens)]
    if eos_token_id in committed:
        committed = committed[: committed.index(eos_token_id) + 1]
    tokens.extend(committed)
    return len(tokens) == limit or (committed and committed[-1] == eos_token_id)


def validate(prompt, max_new_tokens, temperature):
    if not prompt or max_new_tokens < 0 or temperature < 0:
        raise ValueError("need a prompt and nonnegative length/temperature")


def autoregressive(target, prompt, max_new_tokens=32, *, temperature=0.0, seed=0, eos_token_id=None):
    """Reference: one target forward pass per generated token."""
    validate(prompt, max_new_tokens, temperature)
    target.eval()
    rng = torch.Generator().manual_seed(seed)
    tokens = list(prompt)
    limit = len(prompt) + max_new_tokens
    stats = dict(target_calls=0, cache_hits=0, cache_misses=0)
    while len(tokens) < limit:
        token = sample(probabilities(target, tokens, temperature)[-1], rng)
        stats["target_calls"] += 1
        if append_tokens(tokens, [token], limit, eos_token_id):
            break
    return tokens, stats


def generate(
    target,
    draft,
    prompt,
    max_new_tokens=32,
    *,
    lookahead=3,
    temperature=0.0,
    seed=0,
    eos_token_id=None,
):
    """Return (prompt + generated tokens, counters): draft, verify, repeat."""
    validate(prompt, max_new_tokens, temperature)
    if lookahead < 1:
        raise ValueError("lookahead must be positive")
    target.eval()
    draft.eval()
    target_rng = torch.Generator().manual_seed(seed)
    draft_rng = torch.Generator().manual_seed(seed + 1)
    tokens = list(prompt)
    limit = len(prompt) + max_new_tokens
    stats = dict(target_calls=0, cache_hits=0, cache_misses=0)
    while len(tokens) < limit:
        proposal = make_draft(draft, tokens, lookahead, temperature, draft_rng)
        k, bonus = verify(target, tokens, proposal, temperature, target_rng)
        stats["target_calls"] += 1
        if append_tokens(tokens, proposal.tokens[:k] + [bonus], limit, eos_token_id):
            break
    return tokens, stats
