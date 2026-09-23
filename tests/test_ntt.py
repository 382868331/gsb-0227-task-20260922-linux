"""Tests for the hand-written iterative NTT.

Expected values here come from direct modular schoolbook arithmetic (a tiny
local formulation), never from the module under test; the
forward-then-inverse identity is only a supplementary round-trip check.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_ntt.ntt import convolution_mod, next_power_of_two, ntt
from exact_ntt.params import NTT_PRIMES, PRIMITIVE_ROOT


def _schoolbook_mod(a, b, p):
    """Tiny independent modular convolution oracle."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + (x % p) * (y % p)) % p
    return out


class TestNTTRoundTrip(unittest.TestCase):
    def test_inverse_recovers_original(self):
        # Supplementary round-trip consistency check.
        for p in NTT_PRIMES:
            for n in (1, 2, 4, 8, 64, 512):
                vec = [(37 * i * i + 11 * i + 5) % p for i in range(n)]
                got = ntt(ntt(list(vec), p), p, invert=True)
                self.assertEqual(got, vec, msg=f"p={p} n={n}")

    def test_primitive_root_powers_are_roots_of_unity(self):
        # w^(2^k) structure sanity for each prime at length 16.
        for p in NTT_PRIMES:
            w = pow(PRIMITIVE_ROOT, (p - 1) // 16, p)
            self.assertEqual(pow(w, 16, p), 1)
            self.assertNotEqual(pow(w, 8, p), 1)


class TestNTTConvolution(unittest.TestCase):
    def test_matches_schoolbook(self):
        cases = [
            ([1], [1]),
            ([1, 2], [3, 4]),
            ([1, -2, 3], [-4, 5, -6]),  # negatives reduced modulo p
            ([0, 0, 7, 0], [1, 0, -1, 2, 9]),
            ([998244352], [5]),  # -1 mod the first prime
        ]
        for a, b in cases:
            size = next_power_of_two(len(a) + len(b) - 1)
            for p in NTT_PRIMES:
                with self.subTest(p=p, a=a, b=b):
                    got = convolution_mod([x for x in a], [x for x in b], size, p)
                    want = _schoolbook_mod(a, b, p)
                    self.assertEqual(got[: len(want)], want)
                    self.assertTrue(all(x == 0 for x in got[len(want):]))

    def test_bad_lengths_rejected(self):
        with self.assertRaises(ValueError):
            ntt([0, 0, 0], NTT_PRIMES[0])
        with self.assertRaises(ValueError):
            ntt([], NTT_PRIMES[0])

    def test_next_power_of_two(self):
        self.assertEqual(next_power_of_two(1), 1)
        self.assertEqual(next_power_of_two(2), 2)
        self.assertEqual(next_power_of_two(3), 4)
        self.assertEqual(next_power_of_two(511), 512)
        self.assertEqual(next_power_of_two(512), 512)


if __name__ == "__main__":
    unittest.main()
