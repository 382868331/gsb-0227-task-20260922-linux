"""Tests for modulus selection and CRT reconstruction."""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_ntt.crt import garner_extend, reconstruct, select_moduli
from exact_ntt.errors import ExactRecoveryError
from exact_ntt.params import NTT_PRIMES

P1, P2, P3 = NTT_PRIMES
PRODUCT_ALL = P1 * P2 * P3


class TestSelectModuli(unittest.TestCase):
    def test_single_modulus_for_small_bounds(self):
        # 998244353 > 2*bound  <=>  bound <= 499122176
        self.assertEqual(select_moduli(0), (P1,))
        self.assertEqual(select_moduli(1), (P1,))
        self.assertEqual(select_moduli(P1 // 2), (P1,))  # P1 odd: 2*floor(P1/2) = P1-1

    def test_two_moduli_switch_point(self):
        # At bound = (P1+1)//2, 2*bound >= P1, so the first prime is not enough.
        self.assertEqual(select_moduli(P1 // 2 + 1), (P1, P2))
        p12 = P1 * P2
        self.assertEqual(select_moduli(p12 // 2), (P1, P2))
        self.assertEqual(select_moduli(p12 // 2 + 1), (P1, P2, P3))

    def test_max_contract_bounds_use_two_primes(self):
        # Worst in-contract bound: 256 * 10**6 * 10**6 = 2.56e14.
        # P1 alone (~9.98e8) is far too small; P1*P2 (~1.00e18) suffices,
        # so under the public coefficient contract the third prime is never
        # required (it is exercised separately via the real NTT/CRT path).
        worst = 256 * 10**12
        moduli = select_moduli(worst)
        self.assertEqual(moduli, (P1, P2))
        product = P1 * P2
        self.assertGreater(product, 2 * worst)

    def test_three_primes_selected_only_beyond_two_prime_range(self):
        p12 = P1 * P2
        self.assertEqual(select_moduli(p12 // 2 + 1), NTT_PRIMES)

    def test_impossibility_is_proved_not_budgeted(self):
        too_big = PRODUCT_ALL // 2 + 1
        with self.assertRaises(ExactRecoveryError):
            select_moduli(too_big)
        # The boundary itself is still exactly representable:
        self.assertEqual(select_moduli(PRODUCT_ALL // 2), NTT_PRIMES)

    def test_negative_bound_rejected(self):
        with self.assertRaises(ValueError):
            select_moduli(-1)


class TestCRTReconstruction(unittest.TestCase):
    def _residues(self, x, primes):
        return [x % p for p in primes]

    def test_symmetric_extremes_single_prime(self):
        primes = (P1,)
        for x in (0, 1, -1, P1 // 2, -(P1 // 2)):
            self.assertEqual(reconstruct(self._residues(x, primes), primes, P1 // 2), x)

    def test_symmetric_extremes_three_primes(self):
        bound = PRODUCT_ALL // 2
        # Check exact boundary values of the symmetric interval.
        for x in (0, 1, -1, bound, -bound):
            got = reconstruct(self._residues(x, NTT_PRIMES), NTT_PRIMES, bound)
            self.assertEqual(got, x)

    def test_random_integers_match(self):
        rng = random.Random(20260923)
        bound = 10**14
        for _ in range(200):
            x = rng.randrange(-bound, bound + 1)
            primes = select_moduli(bound)
            got = reconstruct(self._residues(x, primes), primes, bound)
            self.assertEqual(got, x)

    def test_garner_extend_agrees_with_brute_crt(self):
        rng = random.Random(4242)
        for _ in range(50):
            x = rng.randrange(-(P1 * P2) // 2, (P1 * P2) // 2 + 1)
            t = x % P1
            t2 = garner_extend(t, P1, x % P2, P2)
            self.assertEqual(t2 % P1, x % P1)
            self.assertEqual(t2 % P2, x % P2)
            self.assertIn(t2, range(P1 * P2))

    def test_out_of_bound_coefficient_rejected(self):
        # Residues whose symmetric representative lies outside the *claimed*
        # bound must be reported, never silently returned.  With one prime
        # the representative is P1//2; a caller that only proved bound 100
        # cannot accept it.  (The normal pipeline proves a bound consistent
        # with the moduli, so this guard is a defensive last line.)
        primes = (P1,)
        huge = P1 // 2
        with self.assertRaises(ExactRecoveryError):
            reconstruct(self._residues(huge, primes), primes, 100)

    def test_insufficient_modulus_rejected(self):
        # Claimed bound not covered even by the symmetric modulus range.
        primes = (P1,)
        with self.assertRaises(ExactRecoveryError):
            reconstruct([0], primes, P1)  # 2*bound == P1, need strictly greater


if __name__ == "__main__":
    unittest.main()
