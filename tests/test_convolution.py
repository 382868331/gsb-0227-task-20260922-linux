"""convolve 端到端测试：期望值全部来自独立朴素参考 tests.reference。"""

import random
import unittest

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    MODULI,
    CoefficientBoundError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    UnprovableBoundError,
    convolve,
    plan_moduli,
)
from exact_conv.ntt import convolution_mod

from tests.reference import naive_convolve


def _assert_matches_naive(test, a, b):
    """核心结果必须逐系数等于独立朴素参考，并检查公开元数据不变量。"""
    result = convolve(a, b)
    expected = naive_convolve(a, b)
    test.assertEqual(result.coefficients, expected)
    test.assertEqual(len(result.coefficients), len(a) + len(b) - 1)
    n, m = len(a), len(b)
    bound = min(n, m) * max(map(abs, a)) * max(map(abs, b))
    test.assertEqual(result.coefficient_bound, bound)
    product = 1
    for mod in result.moduli:
        product *= mod
    test.assertEqual(result.modulus_product, product)
    test.assertGreater(product, 2 * bound)
    # 变换长度必须是 2 的幂且装得下完整卷积。
    size = result.transform_length
    test.assertTrue(size & (size - 1) == 0 and size >= len(expected))
    return result


class CorrectnessTests(unittest.TestCase):
    def test_tiny_hand_picked(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        result = _assert_matches_naive(self, a, b)
        self.assertEqual(result.coefficients, [4, 13, 28, 27, 18])

    def test_all_negative(self):
        # 全负 × 全负 -> 全正；全负 × 正 -> 全负。
        a = [-1, -2, -3, -4]
        b = [-5, -6, -7]
        _assert_matches_naive(self, a, b)
        c = [5, 6, 7]
        result = _assert_matches_naive(self, a, c)
        # 全负 × 全正且无零系数：每个卷积系数严格为负。
        self.assertTrue(all(x < 0 for x in result.coefficients))
        self.assertEqual(result.coefficients[0], -5)

    def test_zeros(self):
        # 全零、被零乘、以及系数中的零散点。
        self.assertEqual(convolve([0, 0, 0], [1, 2, 3]).coefficients, [0] * 5)
        self.assertEqual(convolve([1, 2, 3], [0, 0]).coefficients, [0] * 4)
        _assert_matches_naive(self, [0], [0])
        _assert_matches_naive(self, [1, 0, 0, 0, 2], [0, 3, 0, 0, 0, 7])

    def test_long_tail_of_zeros(self):
        # 长度 256 但有效系数只有前几项：卷积长 511（补零到 512 点 NTT），
        # 尾部大片零必须精确还原，不能因补零/截断错位。
        a = [1, -2, 3] + [0] * (MAX_POLY_LENGTH - 3)
        b = [4, 5] + [0] * (MAX_POLY_LENGTH - 2)
        result = _assert_matches_naive(self, a, b)
        self.assertEqual(len(result.coefficients), 511)
        # k>=4 的所有系数都应为 0。
        self.assertTrue(all(x == 0 for x in result.coefficients[4:]))
        self.assertEqual(result.coefficients[:4], [4, -3, 2, 15])

    def test_length_one_factors(self):
        _assert_matches_naive(self, [10 ** 6], [-(10 ** 6)])
        _assert_matches_naive(self, [255], list(range(1, 257)))

    def test_deterministic_random_matrix(self):
        rng = random.Random(20260923)
        cases = [
            (1, 1, 10 ** 6),
            (2, 3, 10 ** 6),
            (7, 5, 1000),
            (31, 33, 12345),
            (64, 64, 10 ** 6),
            (100, 57, 10 ** 5),
            (255, 256, 10 ** 6),
            (256, 256, 10 ** 6),
        ]
        for n, m, hi in cases:
            a = [rng.randrange(-hi, hi + 1) for _ in range(n)]
            b = [rng.randrange(-hi, hi + 1) for _ in range(m)]
            _assert_matches_naive(self, a, b)

    def test_max_size_max_magnitude(self):
        # 接口上限：256 项、系数绝对值 10^6。最坏系数界
        # B = 256 * 1e6 * 1e6 = 2.56e14，单模（约 1e9）不够、必须双模。
        # 全 +1e6 × 全 -1e6：峰值系数恰为 -256*1e12 = -2.56e14，
        # 压到 CRT 负向恢复与对称区间折叠。
        a = [MAX_COEFF_ABS] * MAX_POLY_LENGTH
        b = [-MAX_COEFF_ABS] * MAX_POLY_LENGTH
        result = _assert_matches_naive(self, a, b)
        self.assertEqual(len(result.moduli), 2)
        # 峰值 2.56e14，远超任一单模数（约 1e9），证明 CRT 真正生效。
        self.assertEqual(max(map(abs, result.coefficients)), 256 * 10 ** 12)
        self.assertGreater(max(map(abs, result.coefficients)), 10 ** 14)

    def test_inputs_are_not_mutated(self):
        # 纯函数：失败或成功都不得改动输入（无部分变更语义）。
        a, b = [1, 2, 3], [4, 5]
        a_snapshot, b_snapshot = list(a), list(b)
        convolve(a, b)
        self.assertEqual((a, b), (a_snapshot, b_snapshot))
        with self.assertRaises(ExactConvolutionError):
            convolve(a, [1, 10 ** 6 + 1])
        self.assertEqual(a, a_snapshot)

    def test_generators_and_tuples_accepted(self):
        result = convolve(iter([1, 2, 3]), (4, 5))
        self.assertEqual(result.coefficients, naive_convolve([1, 2, 3], [4, 5]))


class ModulusSelectionTests(unittest.TestCase):
    def test_single_modulus_case(self):
        # B = 16 * 1000 * 1000 = 1.6e7 < 1004535809/2，单模足够。
        result = convolve([1000] * 16, [-1000] * 16)
        self.assertEqual(result.moduli, (1004535809,))
        self.assertEqual(result.coefficients, naive_convolve([1000] * 16, [-1000] * 16))

    def test_two_modulus_case(self):
        # B = 64 * 1e12 = 6.4e13，单模不够、双模绰绰有余。
        rng = random.Random(31337)
        a = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(64)]
        b = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(64)]
        result = _assert_matches_naive(self, a, b)
        self.assertEqual(result.moduli, (1004535809, 998244353))

    def test_plan_moduli_thresholds(self):
        # 直接检验最少模数选择的严格阈值（纯数学规划，无浮点）。
        p1 = MODULI[0]
        # 界 B=(p1-1)/2 -> 2*B=p1-1 < p1，单模即可。
        b_low = (p1 - 1) // 2
        moduli, bound, product = plan_moduli(b_low, b_low, 1, 1)
        self.assertEqual(moduli, (p1,))
        self.assertEqual(bound, b_low)
        # 界 B=(p1+1)/2 -> 2*B=p1+1 > p1，必须升级双模。
        b_high = (p1 + 1) // 2
        moduli, bound, product = plan_moduli(b_high, b_high, 1, 1)
        self.assertEqual(len(moduli), 2)
        self.assertGreater(product, 2 * bound)

    def test_plan_moduli_zero_bound_uses_one(self):
        # 零输入界 B=0：一个模数即满足 product > 0。
        moduli, bound, product = plan_moduli(256, 256, 0, 9)
        self.assertEqual(moduli, (MODULI[0],))
        self.assertEqual(bound, 0)

    def test_all_three_moduli_path_exists(self):
        # 接口上限内最坏界 2.56e17 只需双模；三模路径由规划函数直接覆盖，
        # 证明“固定三个模数”的第三档确实可用且阈值正确。
        p1, p2, p3 = MODULI
        two_product = p1 * p2
        # 2*B 略大于双模乘积 -> 必须三模（两参数同值以通过 min(n,m)）。
        n = two_product // 2 + 1
        moduli, bound, product = plan_moduli(n, n, 1, 1)
        self.assertEqual(moduli, (p1, p2, p3))
        self.assertEqual(product, two_product * p3)
        self.assertGreater(product, 2 * bound)

    def test_unprovable_when_all_moduli_insufficient(self):
        # 三模乘积仍不够：明确区分“证明无解”（UnprovableBoundError）。
        total_product = 1
        for p in MODULI:
            total_product *= p
        n = total_product // 2 + 1
        with self.assertRaises(UnprovableBoundError):
            plan_moduli(n, n, 1, 1)

    def test_plan_moduli_rejects_bad_scalar_arguments(self):
        # 整数参数拒绝 bool（即使它能当 int 用）、float、非数值与负数。
        for kwargs in (
                dict(n=True, m=1, max_abs_a=1, max_abs_b=1),
                dict(n=1.0, m=1, max_abs_a=1, max_abs_b=1),
                dict(n=1, m="1", max_abs_a=1, max_abs_b=1),
                dict(n=-1, m=1, max_abs_a=1, max_abs_b=1),
                dict(n=1, m=1, max_abs_a=float("nan"), max_abs_b=1),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ExactConvolutionError):
                    plan_moduli(**kwargs)


class MultiModulusAgreementTests(unittest.TestCase):
    def test_residues_agree_with_each_modulus(self):
        # 双模情形下，恢复结果逐系数对每个使用模数取模都必须等于该模 NTT
        # 卷积余数（这是 CRT 正确性的直接、独立核验）。
        rng = random.Random(424242)
        a = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(80)]
        b = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(70)]
        result = convolve(a, b)
        size = result.transform_length
        for mod in result.moduli:
            residues = convolution_mod(a, b, size, mod)
            for got, r in zip(result.coefficients, residues):
                self.assertEqual(got % mod, r)


class InputValidationTests(unittest.TestCase):
    def _assert_error(self, a, b, error_type):
        with self.assertRaises(error_type):
            convolve(a, b)

    def test_bool_coefficients_rejected(self):
        # bool 虽是 int 子类，契约明确拒绝（True/False 都拒绝）。
        self._assert_error([True, 2], [3, 4], InvalidCoefficientError)
        self._assert_error([1, 2], [False], InvalidCoefficientError)

    def test_float_nan_infinity_rejected(self):
        self._assert_error([float("nan"), 1], [2], InvalidCoefficientError)
        self._assert_error([float("inf"), 1], [2], InvalidCoefficientError)
        self._assert_error([1], [float("-inf"), 2], InvalidCoefficientError)
        # 普通有限 float 同样拒绝（接口只接受精确整数）。
        self._assert_error([1.0], [2], InvalidCoefficientError)

    def test_non_numeric_coefficients_rejected(self):
        self._assert_error(["1", 2], [3], InvalidCoefficientError)
        self._assert_error([1, 2], [None, 3], InvalidCoefficientError)
        self._assert_error([1 + 2j], [3], InvalidCoefficientError)

    def test_non_iterable_and_string_rejected(self):
        self._assert_error(123, [1, 2], InvalidPolynomialError)
        self._assert_error([1, 2], None, InvalidPolynomialError)
        # 字符串可迭代但显然不是系数序列，必须在结构层拒绝而非逐字符处理。
        self._assert_error("abc", [1, 2], InvalidPolynomialError)

    def test_empty_and_too_long_rejected(self):
        self._assert_error([], [1, 2], LengthOutOfBoundError)
        self._assert_error([1], [], LengthOutOfBoundError)
        self._assert_error([1] * (MAX_POLY_LENGTH + 1), [1], LengthOutOfBoundError)

    def test_coefficient_bound_rejected_not_reduced(self):
        # 超可证界必须拒绝，而不是取模冒充精确：10^6+1 与 -(10^6+1)。
        self._assert_error([10 ** 6 + 1], [1], CoefficientBoundError)
        self._assert_error([1], [-(10 ** 6 + 1)], CoefficientBoundError)
        # 边界值本身合法。
        result = convolve([MAX_COEFF_ABS], [-MAX_COEFF_ABS])
        self.assertEqual(result.coefficients, [-(10 ** 12)])

    def test_error_messages_are_locatable(self):
        # 错误信息需带出参数名与下标，便于定位。
        with self.assertRaises(InvalidCoefficientError) as ctx:
            convolve([1, 2, "bad"], [3])
        self.assertIn("a[2]", str(ctx.exception))
        with self.assertRaises(CoefficientBoundError) as ctx:
            convolve([1], [0, 10 ** 6 + 1])
        self.assertIn("b[1]", str(ctx.exception))

    def test_errors_are_value_errors(self):
        for bad_a, bad_b, etype in [
                ([True], [1], InvalidCoefficientError),
                ([], [1], LengthOutOfBoundError),
                ([10 ** 6 + 1], [1], CoefficientBoundError),
                (object, [1], InvalidPolynomialError)]:
            self.assertTrue(issubclass(etype, ValueError))
            with self.assertRaises(ValueError):
                convolve(bad_a, bad_b)


if __name__ == "__main__":
    unittest.main()
