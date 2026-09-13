"""Step 2: overlap verification with drafts for predicted verification outcomes.

This implements the SSD framework (arXiv:2603.03251, Algorithm 1), using a simple
top-k outcome cache and the same exact verification as ordinary SD in sd.py.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from sd import append_tokens, make_draft, probabilities, validate, verify


def prepare_cache(draft, prefix, current, lookahead, fanout, temperature, rng):
    # Predict outcomes using only the drafter: (accepted count, bonus token).
    # Score all prefixes, including full acceptance. Keep soft probabilities
    # for ranking alternatives even when the actual proposals are greedy.
    outcome_probs = probabilities(draft, prefix + current.tokens, temperature or 1.0)[len(prefix) - 1 :]
    cache = {}
    for k, q in enumerate(outcome_probs):
        scores = q.clone()
        count = min(fanout, len(scores))
        if k < len(current.tokens):
            # A rejected token has no mass in the residual distribution.
            scores[current.tokens[k]] = -1
            count = min(count, len(scores) - 1)
        for bonus in scores.topk(count).indices.tolist():
            branch_prefix = prefix + current.tokens[:k] + [bonus]
            cache[k, bonus] = make_draft(draft, branch_prefix, lookahead, temperature, rng)
    return cache


def generate(
    target,
    draft,
    prompt,
    max_new_tokens=32,
    *,
    lookahead=3,
    fanout=2,
    temperature=0.0,
    seed=0,
    eos_token_id=None,
):
    """Return (prompt + generated tokens, counters), with concurrent drafting.

    Separate CPU RNGs keep verification independent of speculative branch draws.
    Put models on separate GPUs for device parallelism; CPU threads also work.
    """
    validate(prompt, max_new_tokens, temperature)
    if lookahead < 1 or fanout < 1:
        raise ValueError("lookahead and fanout must be positive")
    target.eval()
    draft.eval()
    target_rng = torch.Generator().manual_seed(seed)
    draft_rng = torch.Generator().manual_seed(seed + 1)
    tokens = list(prompt)
    limit = len(prompt) + max_new_tokens
    stats = dict(target_calls=0, cache_hits=0, cache_misses=0)
    current = None

    with ThreadPoolExecutor(max_workers=1) as worker:
        while len(tokens) < limit:
            if current is None:
                current = make_draft(draft, tokens, lookahead, temperature, draft_rng)
            future = None
            if limit - len(tokens) > 1:
                # Snapshot the prefix: the worker must not see later commits.
                future = worker.submit(
                    prepare_cache, draft, tokens.copy(), current,
                    lookahead, fanout, temperature, draft_rng,
                )
            k, bonus = verify(target, tokens, current, temperature, target_rng)
            stats["target_calls"] += 1
            if append_tokens(tokens, current.tokens[:k] + [bonus], limit, eos_token_id):
                if future is not None:
                    future.result()
                break

            cache = future.result()  # Finish draft work before any fallback.
            current = cache.get((k, bonus))
            stats["cache_hits" if current is not None else "cache_misses"] += 1
            # A miss leaves current=None: next iteration drafts synchronously.
    # Join unused final-round work too; no worker leaks past return.
    return tokens, stats
