"""CPU-ready demo: tiny random causal Transformers, no downloads or training."""

import argparse
import copy
import time

import torch
from torch import nn

import sd
import ssd


class TinyLM(nn.Module):
    def __init__(self, vocab_size, layers=3, width=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2,
                                       dropout=0, batch_first=True)
            for _ in range(layers)
        ])
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.embedding(tokens)
        # Fixed sinusoidal positions; no maximum context setting needed.
        positions = torch.arange(length, device=tokens.device)[:, None]
        frequencies = 10000 ** (-torch.arange(0, x.shape[-1], 2, device=tokens.device) / x.shape[-1])
        angles = positions * frequencies
        x = x + torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        mask = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, src_mask=mask)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--target-device", default="cpu")
    parser.add_argument("--draft-device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    alphabet = " abcdefghijklmnopqrstuvwxyz."
    target = TinyLM(len(alphabet)).eval()
    draft = copy.deepcopy(target)
    draft.blocks = draft.blocks[:1]  # Cheap draft shares initialization, not storage.
    target.to(args.target_device)
    draft.to(args.draft_device)
    prompt = [alphabet.index(c) for c in "the cat"]

    # Warm up both models so initial operator setup is outside the timers.
    with torch.inference_mode():
        for model in (target, draft):
            device = next(model.parameters()).device
            model(torch.tensor([prompt], device=device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    outputs = {}
    print("Random, untrained models: output is meaningless; timings are not a speedup claim.")
    for mode in ("ar", "sd", "ssd"):
        started = time.perf_counter()
        if mode == "ar":
            output, stats = sd.autoregressive(target, prompt, args.tokens, temperature=args.temperature)
        else:
            generate = sd.generate if mode == "sd" else ssd.generate
            options = {} if mode == "sd" else {"fanout": args.fanout}
            output, stats = generate(
                target, draft, prompt, args.tokens, lookahead=args.lookahead,
                temperature=args.temperature, **options,
            )
        elapsed = time.perf_counter() - started
        outputs[mode] = output
        print(f"{mode:3} {elapsed:.3f}s  {stats}  {''.join(alphabet[t] for t in output)!r}")
    if args.temperature == 0:
        assert outputs["ar"] == outputs["sd"] == outputs["ssd"]
        print("Greedy outputs match exactly.")


if __name__ == "__main__":
    main()
