"""Correctness checks on small models whose next-token laws are known exactly."""

from collections import Counter
import threading
import unittest
from unittest.mock import patch

import torch
from torch import nn

import sd
import ssd


class MarkovModel(nn.Module):
    """The last token selects a row of a fixed transition matrix."""

    def __init__(self, transitions):
        super().__init__()
        self.logits = nn.Parameter(
            torch.tensor(transitions, dtype=torch.float64).log(),
            requires_grad=False,
        )
        self.calls = 0

    def forward(self, token_ids):
        self.calls += 1
        return self.logits[token_ids]


CYCLE = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]


class DecodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def run_mode(self, mode, target, draft, prompt, count, **kwargs):
        if mode == "ar":
            return sd.autoregressive(target, prompt, count, **kwargs)
        generator = sd.generate if mode == "sd" else ssd.generate
        return generator(target, draft, prompt, count, **kwargs)

    def test_greedy_first_partial_and_full_acceptance(self):
        cases = [
            ("first rejection", [[0, 0, 1], [0, 0, 1], [1, 0, 0]], 0, 1),
            ("partial acceptance", [[0, 1, 0], [1, 0, 0], [1, 0, 0]], 1, 2),
            ("full acceptance", CYCLE, 2, 0),
        ]
        for name, transitions, expected_k, expected_bonus in cases:
            with self.subTest(case=name):
                target, draft = MarkovModel(CYCLE), MarkovModel(transitions)
                proposal = sd.make_draft(draft, [0], 2, 0.0, torch.Generator())
                outcome = sd.verify(target, [0], proposal, 0.0, torch.Generator())
                self.assertEqual(outcome, (expected_k, expected_bonus))
                self.assertEqual(target.calls, 1)
                for mode in ("ar", "sd", "ssd"):
                    output, _ = self.run_mode(mode, target, draft, [0], 7)
                    self.assertEqual(output, [0, 1, 2, 0, 1, 2, 0, 1], mode)

    def test_equal_sampling_distributions_accept_every_proposal(self):
        transitions = [[0.2, 0.3, 0.5]] * 3
        target, draft = MarkovModel(transitions), MarkovModel(transitions)
        for seed in range(12):
            proposal = sd.make_draft(draft, [0], 3, 1.0, torch.Generator().manual_seed(seed))
            self.assertEqual(proposal.probs.device.type, "cpu")
            torch.testing.assert_close(
                proposal.probs, torch.tensor(transitions, dtype=torch.float64)
            )
            before = target.calls
            k, bonus = sd.verify(target, [0], proposal, 1.0, torch.Generator().manual_seed(seed + 20))
            self.assertEqual(k, 3)
            self.assertIn(bonus, (0, 1, 2))
            self.assertEqual(target.calls - before, 1)

    def test_disjoint_support_uses_the_correction_distribution(self):
        target = MarkovModel([[0, 1, 0]] * 3)
        draft = MarkovModel([[1, 0, 0]] * 3)
        proposal = sd.make_draft(draft, [2], 2, 1.0, torch.Generator())
        self.assertEqual(
            sd.verify(target, [2], proposal, 1.0, torch.Generator()), (0, 1)
        )
        self.assertEqual(target.calls, 1)

    def test_cache_keys_identify_the_exact_continuation_prefix(self):
        draft = MarkovModel([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.3, 0.5, 0.2]])
        current = sd.Draft(
            [0, 1], torch.tensor([[0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
        )
        prefix = [2]

        def record_prefix(model, prefix, lookahead, temperature, rng):
            return sd.Draft(list(prefix), torch.empty(0, 3))

        with patch("ssd.make_draft", side_effect=record_prefix):
            cache = ssd.prepare_cache(draft, prefix, current, 2, 2, 1.0, torch.Generator())
        self.assertEqual(set(cache), {(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 2)})
        for (k, bonus), proposal in cache.items():
            self.assertEqual(proposal.tokens, prefix + current.tokens[:k] + [bonus])
        self.assertEqual(draft.calls, 1)  # Prediction after all current tokens.
        self.assertEqual(prefix, [2])
        self.assertEqual(current.tokens, [0, 1])

    def test_cache_hit_and_miss_preserve_the_greedy_result(self):
        expected = [0, 1, 2, 0, 1, 2, 0, 1]
        target, draft = MarkovModel(CYCLE), MarkovModel(CYCLE)
        output, stats = ssd.generate(target, draft, [0], 7, lookahead=2, fanout=1)
        self.assertEqual(output, expected)
        self.assertGreater(stats["cache_hits"], 0)
        self.assertEqual(stats["target_calls"], target.calls)

        target, draft = MarkovModel(CYCLE), MarkovModel(CYCLE)
        with patch("ssd.prepare_cache", return_value={}):
            output, stats = ssd.generate(target, draft, [0], 7, lookahead=2, fanout=1)
        self.assertEqual(output, expected)
        self.assertGreater(stats["cache_misses"], 0)
        self.assertEqual(stats["cache_hits"], 0)
        self.assertEqual(stats["target_calls"], target.calls)

    def test_eos_and_token_budget_cut_off_a_committed_block(self):
        for mode in ("ar", "sd", "ssd"):
            target, draft = MarkovModel(CYCLE), MarkovModel(CYCLE)
            with self.subTest(mode=mode, case="eos"):
                output, _ = self.run_mode(mode, target, draft, [0], 8, eos_token_id=2)
                self.assertEqual(output, [0, 1, 2])
            for count in (0, 1, 2, 5):
                with self.subTest(mode=mode, count=count):
                    prompt = [0]
                    output, stats = self.run_mode(mode, target, draft, prompt, count)
                    self.assertEqual(output, [0, 1, 2, 0, 1, 2][: count + 1])
                    self.assertEqual(prompt, [0])
                    if count == 0:
                        self.assertEqual(stats["target_calls"], 0)

    def test_future_drafting_overlaps_verification(self):
        verifying = threading.Event()
        prepared = threading.Event()
        checked = threading.Event()
        real_prepare, real_verify = ssd.prepare_cache, ssd.verify

        def prepare_during_verify(*args, **kwargs):
            if checked.is_set():
                return real_prepare(*args, **kwargs)
            self.assertTrue(verifying.wait(5), "verification never began")
            result = real_prepare(*args, **kwargs)
            self.assertFalse(checked.is_set(), "verification ended before drafting")
            prepared.set()
            return result

        def verify_while_preparing(*args, **kwargs):
            if not checked.is_set():
                verifying.set()
                self.assertTrue(prepared.wait(5), "draft preparation did not overlap verification")
                checked.set()
            return real_verify(*args, **kwargs)

        with patch("ssd.prepare_cache", side_effect=prepare_during_verify), patch(
            "ssd.verify", side_effect=verify_while_preparing
        ):
            output, _ = ssd.generate(
                MarkovModel(CYCLE), MarkovModel(CYCLE), [0], 4, lookahead=1, fanout=1
            )
        self.assertTrue(checked.is_set())
        self.assertEqual(output, [0, 1, 2, 0, 1])

    def test_sampling_preserves_the_two_token_joint_distribution(self):
        transitions = [[0.5, 0.3, 0.2], [0.2, 0.3, 0.5], [0.4, 0.4, 0.2]]
        target = MarkovModel(transitions)
        draft = MarkovModel([[0.3, 0.6, 0.1], [0.55, 0.2, 0.25], [0.2, 0.2, 0.6]])
        expected = {
            (first, second): transitions[0][first] * transitions[first][second]
            for first in range(3) for second in range(3)
        }
        trials = 1600
        for mode, fanout in (("sd", None), ("ssd", 1), ("ssd", 3)):
            counts = Counter()
            totals = Counter()
            for seed in range(trials):
                output, stats = self.run_mode(
                    mode, target, draft, [0], 2,
                    lookahead=2, temperature=1.0, seed=seed,
                    **({"fanout": fanout} if fanout is not None else {}),
                )
                counts[tuple(output[1:])] += 1
                totals.update(stats)
            for pair, probability in expected.items():
                with self.subTest(mode=mode, fanout=fanout, pair=pair):
                    self.assertAlmostEqual(counts[pair] / trials, probability, delta=0.045)
            if mode == "ssd":
                self.assertGreater(totals["cache_hits"], 0)
                if fanout == 1:
                    self.assertGreater(totals["cache_misses"], 0)
                else:
                    self.assertEqual(totals["cache_misses"], 0)


if __name__ == "__main__":
    unittest.main()
