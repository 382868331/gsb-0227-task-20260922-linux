"""Tests for the stateless modulus-selection planner."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_ntt import (
    NTT_PRIMES,
    ExactRecoveryError,
    convolve_exact,
    integer_convolution_info,
)


class TestConvolutionInfo(unittest.TestCase):
    def test_info_matches_actual_run(self):
        a = [1, 2, -3, 1]
        b = [4, -5, 6]
        info = integer_convolution_info(4, 3, 3, 6)
        result = convolve_exact(a, b)
        self.assertEqual(info["bound"], result.bound)
        self.assertEqual(info["moduli"], result.moduli)
        self.assertEqual(info["transform_length"], result.transform_length)
        self.assertEqual(info["output_length"], 6)
        self.assertEqual(info["modulus_count"], 1)
        self.assertEqual(info["twice_bound"], 108)
        self.assertGreater(info["modulus_product"], info["twice_bound"])

    def test_info_shape_metadata(self):
        info = integer_convolution_info(256, 256, 10**6, 10**6)
        self.assertEqual(info["bound"], 256 * 10**12)
        self.assertEqual(info["moduli"], NTT_PRIMES[:2])
        self.assertEqual(info["output_length"], 511)
        self.assertEqual(info["transform_length"], 512)

    def test_info_rejects_bool_and_non_ints(self):
        for kwargs in (
            dict(length_a=True, length_b=1, max_abs_a=1, max_abs_b=1),
            dict(length_a=1, length_b=1.0, max_abs_a=1, max_abs_b=1),
            dict(length_a=1, length_b=1, max_abs_a=math.nan, max_abs_b=1),
        ):
            with self.assertRaises(TypeError):
                integer_convolution_info(**kwargs)

    def test_info_rejects_bad_ranges(self):
        with self.assertRaises(ValueError):
            integer_convolution_info(0, 1, 1, 1)
        with self.assertRaises(ValueError):
            integer_convolution_info(1, -2, 1, 1)
        with self.assertRaises(ValueError):
            integer_convolution_info(1, 1, -1, 1)

    def test_provable_impossibility_reachable_through_planner(self):
        # A claimed magnitude no combination of the three fixed primes can
        # separate symmetrically -> ExactRecoveryError (proved impossible,
        # not an exhausted search budget).
        product = 1
        for p in NTT_PRIMES:
            product *= p
        too_large = product  # min(n,m)>=1 makes bound >= max_abs
        with self.assertRaises(ExactRecoveryError):
            integer_convolution_info(1, 1, too_large, 1)


if __name__ == "__main__":
    unittest.main()
