# Speculative decoding, then speculative speculative decoding

Two small PyTorch implementations, meant to be read in order:

1. **[`sd.py`](sd.py)**: ordinary speculative decoding (SD). A draft model
   proposes a block, the target verifies it in one causal forward pass, and the
   process repeats from the committed prefix.
2. **[`ssd.py`](ssd.py)**: speculative speculative decoding (SSD). Imports the
   same drafting and verification functions, adding a worker that prepares
   next-round drafts while the target verifies the current block.

Both support greedy decoding (`temperature=0`) and exact speculative sampling
(`temperature>0`). The target remains the authority for every emitted token.

## Run

Python 3.10+ and PyTorch are the only requirements; no model downloads are needed.

```bash
python3 -m pip install -r requirements.txt
python3 demo.py
python3 demo.py --temperature 0.8
python3 -m unittest -v
```

The demo compares target-only autoregressive decoding (AR), SD, and SSD using a
tiny causal Transformer and a one-layer copy as its draft. **These models are
random and untrained: the output is intentionally meaningless.** Greedy outputs
are asserted equal. Sampled outputs may differ, even with the same seed; the
guarantee is equality of distributions, subject to floating-point arithmetic.

The printed counters show target forward passes and SSD cache hits/misses.
Hits/misses count transitions to another round, excluding the terminal round.
Timings include all work, even unused final-round cache preparation.

To put the models on separate GPUs (optional; this path needs two CUDA devices):

```bash
python3 demo.py --target-device cuda:0 --draft-device cuda:1
```

## The difference in the loops

Ordinary SD waits for each phase:

```text
draft current block -> verify -> commit -> draft next block -> verify ...
```

SSD prepares possible futures during verification:

```text
target:  [verify current block] -> outcome (k, bonus) -> [verify next block]
draft:   [prepare cache for possible outcomes] -------> choose cached draft
                                                      or draft on a miss
```

An outcome is the pair **(number of accepted tokens, correction/bonus token)**.
Suppose the current draft is `sat on a chair`. Cache entry `(2, the)` contains a
new draft starting from `prefix + sat on the`; `(1, under)` starts from
`prefix + sat under`. Only the entry matching the target's actual decision is
used, and its tokens still have to be verified in the next round.

`prepare_cache` considers every acceptance length, including full acceptance,
and up to `fanout` likely bonus tokens at each length. It excludes the rejected
token itself, which cannot be the residual correction. It stores both proposed
tokens and their sampling probabilities. Separate random generators prevent
branch preparation from sharing the verifier's random stream.

There is **no third model**: a cache miss simply drafts synchronously using the
same draft model. The worker receives a snapshot of the prefix, and its work is
joined before fallback or return.

## What verification does

In `sd.verify`, `p` is the target distribution and `q` is the actual proposal
distribution. A proposed token `x` is accepted with probability `min(1, p[x]/q[x])`.
At the first rejection, the correction is sampled from normalized `max(p-q, 0)`;
later draft tokens are discarded. If all proposals survive, an extra token is
sampled from the target after the whole block. Setting temperature to zero makes
both distributions one-hot, recovering greedy prefix matching.

The important indexing is `logits[len(prefix)-1:]`: the position **before** a
token predicts it. One target call supplies all proposal scores plus the final
bonus-token distribution.

## Deliberate simplifications

- One request at a time, shared token IDs, models returning `[batch, time, vocab]`
  logits. To use another PyTorch model, supply that interface and a nonempty prompt.
- No KV caching: full prefixes are recomputed, making rollback unnecessary.
- One draft worker builds branches sequentially, concurrently with target
  verification. The paper's batched branch construction is omitted.
- A simple fixed fanout replaces Saguaro's optimized cache allocation; there is
  no Saguaro proposal-sampling optimization or specialized fallback model.
- CPU threads run the mechanism locally. This is a replication of the **SSD
  framework**, not a reproduction of the paper's performance results: SSD can
  be much slower here. Separate GPUs are supported but were not available for
  validation in the development environment.

The tests use small transition-table models with known distributions to check
acceptance/correction, cache prefixes, hits and fallback, EOS and output limits,
joint sample probabilities, and concurrency using synchronization events.

## Papers

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
  and [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
  describe the exact sampling rule used by `sd.py`.
- [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251), Algorithm 1,
  supplies the outcome-cache and concurrent execution structure used by `ssd.py`.
