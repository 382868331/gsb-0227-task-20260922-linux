"""End-to-end exact convolution tests against the INDEPENDENT naive oracle.

Expected values come exclusively from ``tests.reference.naive_convolution``
(plain schoolbook arithmetic); the NTT core is never used to derive them.
Covers the required shapes: all-negative, zero, long zero tails, and the
single-/multi-modulus switch points.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_ntt import MAX_COEFF_ABS, NTT_PRIMES, convolve_exact
from exact_ntt.crt import reconstruct
from exact_ntt.ntt import convolution_mod, next_power_of_two

from reference import naive_convolution


class TestExactConvolution(unittest.TestCase):
    def _assert_exact(self, a, b, expected_modulus_count=None):
        result = convolve_exact(a, b)
        want = naive_convolution(a, b)
        self.assertEqual(result.coefficients, want)
        self.assertEqual(len(result.coefficients), len(a) + len(b) - 1)
        # Proof metadata must actually prove the result.
        self.assertGreater(result.modulus_product, 2 * result.bound)
        self.assertEqual(result.transform_length,
                         next_power_of_two(len(a) + len(b) - 1))
        if expected_modulus_count is not None:
            self.assertEqual(len(result.moduli), expected_modulus_count)
        return result

    def test_tiny_single_modulus(self):
        result = self._assert_exact([1, 2, 3], [4, -5], expected_modulus_count=1)
        self.assertEqual(result.moduli, (NTT_PRIMES[0],))
        self.assertEqual(result.bound, 2 * 3 * 5)

    def test_all_negative(self):
        self._assert_exact([-1, -2, -3, -4], [-5, -6, -7], 1)

    def test_mixed_signs_including_boundary_coefficients(self):
        self._assert_exact([MAX_COEFF_ABS, -MAX_COEFF_ABS, 0, 1],
                           [-MAX_COEFF_ABS, MAX_COEFF_ABS], 2)

    def test_all_zero_operands(self):
        result = self._assert_exact([0, 0, 0], [0, 0], expected_modulus_count=1)
        self.assertEqual(result.bound, 0)
        self.assertTrue(all(c == 0 for c in result.coefficients))

    def test_long_zero_tails(self):
        # Sparse values with long deterministic zero tails, both operands
        # padded out near the length bound.
        a = [7, -3, 11] + [0] * 253          # length 256
        b = [2, 0, 0, -9] + [0] * 252        # length 256
        result = self._assert_exact(a, b, expected_modulus_count=1)
        # Non-zero output exists only in the first few coefficients.
        self.assertTrue(any(c != 0 for c in result.coefficients[:6]))
        self.assertTrue(all(c == 0 for c in result.coefficients[6:]))

    def test_single_to_two_modulus_switch(self):
        # Bound just under the first-prime threshold: min(n,m)*Amax*Bmax
        # = 256 * 1400 * 1400 = 501,760,000 > 998244353/2 = 499,122,176.
        rng = random.Random(7)
        a = [rng.choice([-1400, 0, 1400]) for _ in range(256)]
        b = [rng.choice([-1400, 1400]) for _ in range(256)]
        # Make the bound tight and explicit:
        a[0], b[0] = 1400, 1400
        result = self._assert_exact(a, b, expected_modulus_count=2)
        self.assertEqual(result.bound, 256 * 1400 * 1400)

        # The same shape with smaller magnitudes stays on one modulus.
        a2 = [x // 100 for x in a] or [1]
        a2[0] = 14
        b2 = [14] * 256
        self._assert_exact(a2, b2, expected_modulus_count=1)

    def test_maximum_contract_inputs(self):
        # n = m = 256, |coef| = 10**6: proved bound 2.56e14 -> two moduli.
        rng = random.Random(99)
        a = [rng.choice([-MAX_COEFF_ABS, MAX_COEFF_ABS]) for _ in range(256)]
        b = [rng.choice([-MAX_COEFF_ABS, MAX_COEFF_ABS]) for _ in range(256)]
        result = self._assert_exact(a, b, expected_modulus_count=2)
        self.assertEqual(result.bound, 256 * 10**12)

    def test_determinism_and_immutability(self):
        a, b = [1, -2, 0, 3], [4, 5, -6]
        r1 = convolve_exact(a, b)
        r2 = convolve_exact(a, b)
        self.assertEqual(r1.coefficients, r2.coefficients)
        self.assertEqual(r1, r2)
        self.assertEqual(a, [1, -2, 0, 3])
        self.assertEqual(b, [4, 5, -6])
        # Returned list is owned by the caller.
        r1.coefficients[0] = 10**9
        self.assertEqual(convolve_exact(a, b).coefficients[0], r2.coefficients[0])

    def test_three_modulus_path_through_real_transforms(self):
        # The public input contract (bound <= 2.56e14) never forces the third
        # prime because P1*P2/2 ~ 5.01e17; the third modulus is nonetheless
        # part of the fixed CRT set.  Exercise the real NTT -> 3x CRT
        # composition on a value outside the public coefficient contract,
        # bypassing only validation (using the same primitives the core uses).
        target = 10**20  # > P1*P2/2, requires all three primes
        a, b = [target], [1]
        size = 1
        residues = [convolution_mod(a, b, size, p)[0] for p in NTT_PRIMES]
        bound = target
        self.assertEqual(reconstruct(residues, NTT_PRIMES, bound), target)


if __name__ == "__main__":
    unittest.main()
