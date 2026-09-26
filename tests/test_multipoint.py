"""多点求值与插值测试：对照独立 Horner 与 Vandermonde 消元参考。"""

import random
import unittest

from exact_conv import (
    MAX_DEGREE,
    MAX_POINTS,
    DuplicatePointError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    UnsupportedModulusError,
    evaluate_multi,
    interpolate,
)
from exact_conv.ntt import MODULI

from tests.reference import horner_evaluate, vandermonde_interpolate

P = 998244353


def _trim(coeffs):
    last = len(coeffs) - 1
    while last > 0 and coeffs[last] == 0:
        last -= 1
    return coeffs[: last + 1]


class EvaluateTests(unittest.TestCase):
    def test_hand_computed(self):
        # f = 1 + 2x + 3x^2；f(0)=1, f(1)=6, f(2)=17。
        result = evaluate_multi([1, 2, 3], [0, 1, 2], P)
        self.assertEqual(result.values, [1, 6, 17])
        self.assertEqual(result.modulus, P)
        self.assertGreaterEqual(result.multiplications, 1)

    def test_zero_polynomial(self):
        result = evaluate_multi([0, 0, 0], [5, 7, 9, 11], P)
        self.assertEqual(result.values, [0, 0, 0, 0])

    def test_single_point(self):
        result = evaluate_multi([3, 4, 5], [10], P)
        self.assertEqual(result.values, [horner_evaluate([3, 4, 5], 10, P)])

    def test_point_zero(self):
        result = evaluate_multi([7, 8, 9], [0], P)
        self.assertEqual(result.values, [7])

    def test_degree_higher_than_point_count(self):
        # 次数 9，只有 2 个点。
        coeffs = list(range(1, 11))
        points = [3, 100]
        result = evaluate_multi(coeffs, points, P)
        expect = [horner_evaluate(coeffs, x, P) for x in points]
        self.assertEqual(result.values, expect)

    def test_random_against_horner_all_moduli(self):
        rng = random.Random(20260926)
        for mod in MODULI:
            for _ in range(5):
                deg = rng.randrange(0, 40)
                coeffs = [rng.randrange(mod) for _ in range(deg + 1)]
                points = [rng.randrange(mod) for _ in range(rng.randrange(1, 30))]
                result = evaluate_multi(coeffs, points, mod)
                expect = [horner_evaluate(coeffs, x, mod) for x in points]
                self.assertEqual(result.values, expect)

    def test_negative_and_large_inputs_normalized(self):
        coeffs = [-1, 2 * P + 3, -5]
        points = [-4, P + 10]
        result = evaluate_multi(coeffs, points, P)
        norm_c = [c % P for c in coeffs]
        expect = [horner_evaluate(norm_c, x % P, P) for x in points]
        self.assertEqual(result.values, expect)
        for v in result.values:
            self.assertTrue(0 <= v < P)

    def test_limits_128(self):
        rng = random.Random(7)
        coeffs = [rng.randrange(P) for _ in range(MAX_DEGREE + 1)]
        points = [rng.randrange(P) for _ in range(MAX_POINTS)]
        result = evaluate_multi(coeffs, points, P)
        expect = [horner_evaluate(coeffs, x, P) for x in points]
        self.assertEqual(result.values, expect)
        with self.assertRaises(LengthOutOfBoundError):
            evaluate_multi(coeffs, points + [1], P)
        with self.assertRaises(LengthOutOfBoundError):
            evaluate_multi(coeffs + [1], points, P)

    def test_multiplication_count_positive_and_deterministic(self):
        rng = random.Random(11)
        coeffs = [rng.randrange(P) for _ in range(50)]
        points = [rng.randrange(P) for _ in range(64)]
        r1 = evaluate_multi(coeffs, points, P)
        r2 = evaluate_multi(coeffs, points, P)
        self.assertEqual(r1.multiplications, r2.multiplications)
        # 64 个点的乘积树需要 63 次乘法。
        self.assertEqual(r1.multiplications, 63)

    def test_input_rejection(self):
        with self.assertRaises(UnsupportedModulusError):
            evaluate_multi([1], [1], 97)
        with self.assertRaises(UnsupportedModulusError):
            evaluate_multi([1], [1], True)
        with self.assertRaises(InvalidPolynomialError):
            evaluate_multi(5, [1], P)
        with self.assertRaises(LengthOutOfBoundError):
            evaluate_multi([], [1], P)
        with self.assertRaises(InvalidCoefficientError):
            evaluate_multi([True], [1], P)
        with self.assertRaises(InvalidCoefficientError):
            evaluate_multi([1], [1.5], P)


class InterpolateTests(unittest.TestCase):
    def test_hand_computed(self):
        # 过 (0,1), (1,6), (2,17) 的二次多项式是 1 + 2x + 3x^2。
        result = interpolate([0, 1, 2], [1, 6, 17], P)
        self.assertEqual(result.coefficients, [1, 2, 3])

    def test_single_point(self):
        result = interpolate([42], [123], P)
        self.assertEqual(result.coefficients, [123])
        self.assertEqual(result.multiplications, 0)

    def test_point_zero(self):
        result = interpolate([0, 1], [5, 8], P)
        self.assertEqual(result.coefficients, [5, 3])

    def test_random_against_vandermonde_all_moduli(self):
        rng = random.Random(31415)
        for mod in MODULI:
            for _ in range(5):
                n = rng.randrange(1, 25)
                points = rng.sample(range(mod), n)
                values = [rng.randrange(mod) for _ in range(n)]
                result = interpolate(points, values, mod)
                expect = _trim(vandermonde_interpolate(points, values, mod))
                self.assertEqual(result.coefficients, expect)
                self.assertLess(len(result.coefficients), n + 1)

    def test_coefficients_normalized(self):
        rng = random.Random(99)
        points = rng.sample(range(P), 30)
        values = [rng.randrange(-10**12, 10**12) for _ in range(30)]
        result = interpolate(points, values, P)
        for c in result.coefficients:
            self.assertTrue(0 <= c < P)

    def test_roundtrip_supplementary(self):
        # 往返仅作补充：插值后再多点求值应回到原值。
        rng = random.Random(555)
        n = 40
        points = rng.sample(range(P), n)
        values = [rng.randrange(P) for _ in range(n)]
        poly = interpolate(points, values, P)
        back = evaluate_multi(poly.coefficients, points, P)
        self.assertEqual(back.values, values)

    def test_duplicate_points_rejected(self):
        with self.assertRaises(DuplicatePointError):
            interpolate([1, 2, 1], [1, 2, 3], P)
        # 模 p 意义下重复（1 与 p+1）也必须拒绝。
        with self.assertRaises(DuplicatePointError):
            interpolate([1, P + 1], [1, 2], P)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ExactConvolutionError):
            interpolate([1, 2], [1], P)

    def test_limits_128(self):
        rng = random.Random(2026)
        points = rng.sample(range(P), MAX_POINTS)
        values = [rng.randrange(P) for _ in range(MAX_POINTS)]
        result = interpolate(points, values, P)
        expect = _trim(vandermonde_interpolate(points, values, P))
        self.assertEqual(result.coefficients, expect)
        with self.assertRaises(LengthOutOfBoundError):
            interpolate(points + [P - 1], values + [0], P)

    def test_multiplication_count_deterministic(self):
        rng = random.Random(21)
        points = rng.sample(range(P), 64)
        values = [rng.randrange(P) for _ in range(64)]
        r1 = interpolate(points, values, P)
        r2 = interpolate(points, values, P)
        self.assertEqual(r1.multiplications, r2.multiplications)
        # 64 点：乘积树 63 次 + 合并树每层 2 次/内部节点共 126 次。
        self.assertEqual(r1.multiplications, 63 + 126)

    def test_input_rejection(self):
        with self.assertRaises(UnsupportedModulusError):
            interpolate([1], [1], 97)
        with self.assertRaises(InvalidPolynomialError):
            interpolate("abc", [1], P)
        with self.assertRaises(InvalidCoefficientError):
            interpolate([1], [False], P)


if __name__ == "__main__":
    unittest.main()
