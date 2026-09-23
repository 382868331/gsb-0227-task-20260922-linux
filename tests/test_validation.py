"""Tests for input validation and locatable error semantics."""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_ntt import (
    MAX_COEFF_ABS,
    MAX_LENGTH,
    CoefficientError,
    LengthError_,
    convolve_exact,
)
from exact_ntt.validation import coefficient_bound, validate_operand


class TestValidateOperand(unittest.TestCase):
    def test_accepts_lists_and_tuples_and_copies(self):
        data = [1, -2, 3]
        checked = validate_operand(data, "a")
        self.assertEqual(checked, [1, -2, 3])
        data[0] = 999  # mutating the caller's list must not affect the copy
        self.assertEqual(checked[0], 1)
        self.assertEqual(validate_operand((4, 5), "b"), [4, 5])

    def test_bool_coefficients_rejected(self):
        with self.assertRaises(CoefficientError) as ctx:
            validate_operand([1, True, 3], "a")
        self.assertIn("a[1]", str(ctx.exception))

    def test_float_coefficients_rejected_with_explicit_messages(self):
        for bad in (math.nan, math.inf, -math.inf, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(CoefficientError) as ctx:
                    validate_operand([0, bad], "a")
                self.assertIn("a[1]", str(ctx.exception))
                if math.isnan(bad):
                    self.assertIn("NaN", str(ctx.exception))
                elif math.isinf(bad):
                    self.assertIn("infinity", str(ctx.exception))

    def test_non_numeric_coefficients_rejected(self):
        with self.assertRaises(CoefficientError):
            validate_operand([1, "2"], "a")
        with self.assertRaises(CoefficientError):
            validate_operand([None], "a")

    def test_coefficient_boundary_values_accepted(self):
        self.assertEqual(validate_operand([MAX_COEFF_ABS, -MAX_COEFF_ABS], "a"),
                         [MAX_COEFF_ABS, -MAX_COEFF_ABS])

    def test_coefficient_out_of_range_rejected(self):
        for bad in (MAX_COEFF_ABS + 1, -(MAX_COEFF_ABS + 1)):
            with self.subTest(bad=bad):
                with self.assertRaises(CoefficientError) as ctx:
                    validate_operand([bad], "a")
                self.assertIn("exceeds the proved bound", str(ctx.exception))

    def test_empty_and_oversized_and_wrong_type(self):
        with self.assertRaises(LengthError_):
            validate_operand([], "a")
        with self.assertRaises(LengthError_):
            validate_operand([0] * (MAX_LENGTH + 1), "a")
        with self.assertRaises(LengthError_):
            validate_operand((x for x in (1, 2)), "a")  # generator rejected
        with self.assertRaises(LengthError_):
            validate_operand("123", "a")

    def test_boundary_length_accepted(self):
        self.assertEqual(len(validate_operand([1] * MAX_LENGTH, "a")), MAX_LENGTH)


class TestCoefficientBound(unittest.TestCase):
    def test_bound_formula(self):
        self.assertEqual(coefficient_bound([3, -7], [2, 10, -4]), 2 * 7 * 10)
        self.assertEqual(coefficient_bound([0, 0], [10**6] * 5), 0)
        self.assertEqual(coefficient_bound([1], [-5]), 5)


class TestConvolveValidationIntegration(unittest.TestCase):
    def test_errors_are_locatable_and_value_errors(self):
        with self.assertRaises(CoefficientError):
            convolve_exact([1, True], [1, 1])
        with self.assertRaises(CoefficientError):
            convolve_exact([math.nan], [1])
        with self.assertRaises(LengthError_):
            convolve_exact([], [1])
        with self.assertRaises(CoefficientError):
            convolve_exact([10**6 + 1], [1])

    def test_statelessness_input_not_mutated(self):
        a, b = [1, 2, 3], [4, -5]
        a_before, b_before = list(a), list(b)
        convolve_exact(a, b)
        self.assertEqual(a, a_before)
        self.assertEqual(b, b_before)


if __name__ == "__main__":
    unittest.main()
