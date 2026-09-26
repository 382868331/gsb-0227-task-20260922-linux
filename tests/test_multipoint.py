"""多点求值 / 插值测试：独立 Horner 与 Vandermonde 消元参考。

参考实现只按定义逐点 Horner 求值、以及直接对 Vandermonde 系统做
高斯消元求解，*不导入* exact_conv.multipoint 的任何内部函数，
因此期望值独立于被测核心；往返一致性只作为补充检验。
"""

import random
import unittest

from exact_conv import (
    MAX_DEGREE,
    MAX_POINTS,
    MODULI,
    DuplicatePointError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidModulusError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    interpolate,
    multipoint_evaluate,
)

P = MODULI[0]  # 1004535809
Q = MODULI[1]  # 998244353


def horner_evaluate(coeffs, x, p):
    """独立参考：按定义逐点 Horner，O(degree)。"""
    value = 0
    for coeff in reversed(coeffs):
        value = (value * x + coeff) % p
    return value


def vandermonde_interpolate(points, values, p):
    """独立参考：直接对 Vandermonde 系统做高斯消元，返回 n 个系数。"""
    n = len(points)
    # 构造增广矩阵：行 i 为 [x_i^0, x_i^1, ..., x_i^(n-1) | y_i]
    matrix = []
    for x, y in zip(points, values):
        row = [1] * (n + 1)
        for k in range(1, n):
            row[k] = row[k - 1] * x % p
        row[n] = y % p
        matrix.append(row)
    # 高斯消元（列主元）。
    for col in range(n):
        pivot = next(r for r in range(col, n) if matrix[r][col] % p != 0)
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        inv = pow(matrix[col][col], p - 2, p)
        matrix[col] = [v * inv % p for v in matrix[col]]
        for r in range(n):
            if r != col and matrix[r][col] % p != 0:
                factor = matrix[r][col]
                matrix[r] = [(a - factor * b) % p
                             for a, b in zip(matrix[r], matrix[col])]
    return [matrix[r][n] for r in range(n)]


def trimmed(coeffs):
    end = len(coeffs)
    while end > 1 and coeffs[end - 1] == 0:
        end -= 1
    return coeffs[:end]


class MultipointEvaluateTest(unittest.TestCase):
    def test_matches_horner_random(self):
        rng = random.Random(20260926)
        for _ in range(30):
            degree = rng.randint(0, 40)
            n = rng.randint(1, 30)
            coeffs = [rng.randrange(P) for _ in range(degree + 1)]
            points = [rng.randrange(P) for _ in range(n)]
            result = multipoint_evaluate(coeffs, points, P)
            self.assertEqual(
                result.values,
                [horner_evaluate(coeffs, x, P) for x in points],
            )
            self.assertEqual(result.modulus, P)

    def test_all_supported_moduli(self):
        coeffs = [3, 1, 4, 1, 5, 9, 2, 6]
        points = [0, 1, 2, 100, 10**12]
        for p in MODULI:
            result = multipoint_evaluate(coeffs, points, p)
            self.assertEqual(
                result.values,
                [horner_evaluate(coeffs, x % p, p) for x in points],
            )

    def test_zero_polynomial(self):
        result = multipoint_evaluate([0, 0, 0], [1, 2, 3], P)
        self.assertEqual(result.values, [0, 0, 0])
        self.assertEqual(result.multiplication_count, 0)

    def test_empty_coeffs_is_zero_polynomial(self):
        result = multipoint_evaluate([], [7, 8], P)
        self.assertEqual(result.values, [0, 0])

    def test_single_point(self):
        result = multipoint_evaluate([5, 6, 7], [2], P)
        self.assertEqual(result.values, [horner_evaluate([5, 6, 7], 2, P)])

    def test_point_zero_picks_constant_term(self):
        self.assertEqual(multipoint_evaluate([42, 9, 9], [0], P).values, [42])
        self.assertEqual(multipoint_evaluate([P, 1], [0], P).values, [0])

    def test_degree_higher_than_point_count(self):
        coeffs = list(range(1, 60))  # 次数 58，只有 3 个点
        points = [3, 5, 7]
        result = multipoint_evaluate(coeffs, points, P)
        self.assertEqual(
            result.values, [horner_evaluate(coeffs, x, P) for x in points]
        )

    def test_coefficients_normalized_to_field(self):
        # 大整数与负数系数先规范到 [0, p-1]，不按整数 CRT 语义处理。
        coeffs = [P, -1, 2 * P + 5]
        result = multipoint_evaluate(coeffs, [1], P)
        self.assertEqual(result.values, [(0 + (P - 1) + 5) % P])
        for v in result.values:
            self.assertTrue(0 <= v < P)

    def test_duplicate_points_allowed_for_evaluation(self):
        result = multipoint_evaluate([1, 1], [4, 4, 4], P)
        self.assertEqual(result.values, [5, 5, 5])

    def test_multiplication_count_deterministic_and_positive(self):
        coeffs = list(range(30))
        points = list(range(1, 33))
        first = multipoint_evaluate(coeffs, points, P)
        second = multipoint_evaluate(coeffs, points, P)
        self.assertGreater(first.multiplication_count, 0)
        self.assertEqual(first.multiplication_count, second.multiplication_count)
        # 32 个叶子的满二叉乘积树恰好 31 次合并乘法。
        self.assertEqual(first.multiplication_count, 31)

    def test_boundary_max_points_and_degree(self):
        rng = random.Random(7)
        coeffs = [rng.randrange(P) for _ in range(MAX_DEGREE + 1)]
        points = rng.sample(range(10**6), MAX_POINTS)
        result = multipoint_evaluate(coeffs, points, P)
        self.assertEqual(
            result.values, [horner_evaluate(coeffs, x, P) for x in points]
        )

    def test_rejects(self):
        with self.assertRaises(InvalidModulusError):
            multipoint_evaluate([1], [1], 7)  # 非固定 NTT 素数
        with self.assertRaises(InvalidModulusError):
            multipoint_evaluate([1], [1], True)
        with self.assertRaises(InvalidModulusError):
            multipoint_evaluate([1], [1], 998244353.0)
        with self.assertRaises(LengthOutOfBoundError):
            multipoint_evaluate([1], [], P)
        with self.assertRaises(LengthOutOfBoundError):
            multipoint_evaluate([1], list(range(MAX_POINTS + 1)), P)
        with self.assertRaises(LengthOutOfBoundError):
            multipoint_evaluate([0] * (MAX_DEGREE + 2), [1], P)
        with self.assertRaises(InvalidCoefficientError):
            multipoint_evaluate([1, True], [1], P)
        with self.assertRaises(InvalidCoefficientError):
            multipoint_evaluate([1], [1.5], P)
        with self.assertRaises(InvalidPolynomialError):
            multipoint_evaluate(5, [1], P)


class InterpolateTest(unittest.TestCase):
    def test_matches_vandermonde_random(self):
        rng = random.Random(1227)
        for _ in range(25):
            n = rng.randint(1, 12)
            points = rng.sample(range(500), n)
            values = [rng.randrange(P) for _ in range(n)]
            result = interpolate(points, values, P)
            expected = trimmed(vandermonde_interpolate(points, values, P))
            self.assertEqual(result.coefficients, expected)
            self.assertLessEqual(len(result.coefficients), n)

    def test_all_supported_moduli(self):
        points = [0, 3, 7]
        values = [5, 11, 13]
        for p in MODULI:
            result = interpolate(points, values, p)
            self.assertEqual(
                result.coefficients,
                trimmed(vandermonde_interpolate(points, values, p)),
            )

    def test_single_point_gives_constant(self):
        self.assertEqual(interpolate([9], [123], P).coefficients, [123])
        self.assertEqual(interpolate([0], [0], P).coefficients, [0])

    def test_point_zero(self):
        # 过 (0, c) 的插值常数项必须为 c。
        result = interpolate([0, 1, 2], [10, 11, 14], P)
        self.assertEqual(result.coefficients[0], 10)
        self.assertEqual(
            result.coefficients,
            trimmed(vandermonde_interpolate([0, 1, 2], [10, 11, 14], P)),
        )

    def test_zero_values_give_zero_polynomial(self):
        self.assertEqual(interpolate([1, 2, 3], [0, 0, 0], P).coefficients, [0])

    def test_coefficients_normalized_to_field(self):
        result = interpolate([1, 2], [P + 3, -1], P)
        for c in result.coefficients:
            self.assertTrue(0 <= c < P)
        self.assertEqual(
            result.coefficients,
            trimmed(vandermonde_interpolate([1, 2], [3, P - 1], P)),
        )

    def test_roundtrip_is_supplementary_check(self):
        # 往返（求值后再插值）仅作补充：主对照是 Vandermonde 消元。
        rng = random.Random(99)
        n = 20
        coeffs = [rng.randrange(P) for _ in range(n)]
        points = rng.sample(range(10**5), n)
        values = multipoint_evaluate(coeffs, points, P).values
        result = interpolate(points, values, P)
        self.assertEqual(result.coefficients, coeffs)

    def test_multiplication_count_deterministic(self):
        points = list(range(1, 17))
        values = [i * i for i in points]
        first = interpolate(points, values, P)
        second = interpolate(points, values, P)
        self.assertGreater(first.multiplication_count, 0)
        self.assertEqual(first.multiplication_count, second.multiplication_count)

    def test_boundary_max_points(self):
        rng = random.Random(313)
        points = rng.sample(range(10**6), MAX_POINTS)
        values = [rng.randrange(P) for _ in range(MAX_POINTS)]
        result = interpolate(points, values, P)
        self.assertLessEqual(len(result.coefficients), MAX_POINTS)
        # 抽查若干点的函数值（完整 Vandermonde 在 128 阶下太慢）。
        for i in (0, 63, 127):
            self.assertEqual(
                horner_evaluate(result.coefficients, points[i], P), values[i]
            )

    def test_duplicate_points_rejected(self):
        with self.assertRaises(DuplicatePointError):
            interpolate([1, 2, 1], [1, 2, 3], P)
        with self.assertRaises(DuplicatePointError):
            interpolate([0, P], [1, 2], P)  # 模 p 后相同
        # 求值不拒绝重复点，仅插值拒绝。
        self.assertEqual(multipoint_evaluate([1], [5, 5], P).values, [1, 1])

    def test_rejects(self):
        with self.assertRaises(ExactConvolutionError):
            interpolate([1, 2], [1], P)  # 长度不匹配
        with self.assertRaises(InvalidModulusError):
            interpolate([1], [1], 2)
        with self.assertRaises(InvalidCoefficientError):
            interpolate([1], [False], P)
        with self.assertRaises(LengthOutOfBoundError):
            interpolate([], [], P)


if __name__ == "__main__":
    unittest.main()

