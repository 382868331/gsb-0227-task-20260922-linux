"""cyclic_mul / negacyclic_mul 端到端测试。

期望值全部来自独立朴素参考 tests.reference.naive_ring_mul（双循环按
指数折返），不调用被测核心。
"""

import random
import unittest

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    CoefficientBoundError,
    ExactConvolutionError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    RingProductResult,
    convolve,
    cyclic_mul,
    negacyclic_mul,
)

from tests.reference import naive_convolve, naive_ring_mul


def _assert_result(test, result, expected, n, negative, n_in, m_in,
                   max_a, max_b):
    """逐系数对拍朴素参考，并核验全部公开可证性元数据。"""
    test.assertIsInstance(result, RingProductResult)
    test.assertEqual(result.coefficients, expected)
    test.assertEqual(len(result.coefficients), n)
    test.assertEqual(result.ring_size, n)
    expect_relation = f"x^{n} + 1" if negative else f"x^{n} - 1"
    test.assertEqual(result.relation, expect_relation)
    # 界：线性界 B=min(n_in,m_in)*maxa*maxb；折返界 ceil((n_in+m_in-1)/N)*B。
    linear_bound = min(n_in, m_in) * max_a * max_b
    blocks = (n_in + m_in - 1 + n - 1) // n
    folded_bound = blocks * linear_bound
    test.assertEqual(result.linear_coefficient_bound, linear_bound)
    test.assertEqual(result.coefficient_bound, folded_bound)
    product = 1
    for mod in result.moduli:
        product *= mod
    test.assertEqual(result.modulus_product, product)
    # 模数乘积必须严格覆盖折返后可能相加的绝对界。
    test.assertGreater(product, 2 * folded_bound)
    # 输出系数确实在公布的界内（可核验）。
    test.assertTrue(all(abs(c) <= folded_bound for c in result.coefficients))
    size = result.transform_length
    test.assertTrue(size & (size - 1) == 0 and size >= n_in + m_in - 1)


def _check_pair(test, a, b, n, negative=False):
    func = negacyclic_mul if negative else cyclic_mul
    result = func(a, b, n)
    expected = naive_ring_mul(a, b, n, negative=negative)
    _assert_result(test, result, expected, n, negative,
                   len(a), len(b), max(map(abs, a)), max(map(abs, b)))
    return result


class CyclicCorrectnessTests(unittest.TestCase):
    def test_hand_picked_cyclic(self):
        # N=4: C = [1,3,4,3,1]，折返 R[0]=C[0]+C[4]=2。
        a, b, n = [1, 1, 1], [1, 2, 1], 4
        result = _check_pair(self, a, b, n)
        self.assertEqual(result.coefficients, [2, 3, 4, 3])
        # 线性卷积长度 5 > N=4，确实发生折返（块数 2）。
        self.assertEqual(result.coefficient_bound, 2 * result.linear_coefficient_bound)

    def test_negative_hand_picked(self):
        # 同例负循环：高次块 (x^4) 变号，R[0] = C[0] - C[4] = 1 - 1 = 0。
        a, b, n = [1, 1, 1], [1, 2, 1], 4
        result = _check_pair(self, a, b, n, negative=True)
        self.assertEqual(result.coefficients, [0, 3, 4, 3])

    def test_sign_difference_cyclic_vs_negacyclic(self):
        # 同一对输入：两种结果只在折返系数上相差“两倍高块贡献”。
        rng = random.Random(7777)
        for n in (1, 2, 4, 8, 16, 64, 256):
            la = min(n, 1 + rng.randrange(n))
            lb = min(n, 1 + rng.randrange(n))
            a = [rng.randrange(-500, 501) for _ in range(la)]
            b = [rng.randrange(-500, 501) for _ in range(lb)]
            cyc = cyclic_mul(a, b, n).coefficients
            neg = negacyclic_mul(a, b, n).coefficients
            linear = naive_convolve(a, b)
            expect_cyc = naive_ring_mul(a, b, n, False)
            expect_neg = naive_ring_mul(a, b, n, True)
            self.assertEqual(cyc, expect_cyc)
            self.assertEqual(neg, expect_neg)
            # 差只可能落在有高次块折回的余数上，且差值为 2*高块和。
            for r in range(n):
                high = sum(linear[k] for k in range(n + r, len(linear), n))
                self.assertEqual(cyc[r] - neg[r], 2 * high)

    def test_highest_times_highest(self):
        # 最高次乘最高次：x^(N-1) * x^(N-1) = x^(2N-2)。
        for n in (1, 2, 4, 256):
            a = [0] * (n - 1) + [7] if n > 1 else [7]
            b = [0] * (n - 1) + [3] if n > 1 else [3]
            cyc = cyclic_mul(a, b, n).coefficients
            neg = negacyclic_mul(a, b, n).coefficients
            if n == 1:
                self.assertEqual(cyc, [21])
                self.assertEqual(neg, [21])
            else:
                expect_cyc = [0] * n
                expect_cyc[n - 2] = 21  # 2N-2 = 2*(N-1) 折回余数 N-2
                self.assertEqual(cyc, expect_cyc)
                expect_neg = [0] * n
                expect_neg[n - 2] = -21  # x^(2N-2)=x^(N-2)*x^N -> -x^(N-2)
                self.assertEqual(neg, expect_neg)

    def test_n_equals_one(self):
        # N=1：Z[x]/(x-1) 与 Z[x]/(x+1) 对常数乘法一致（x=±1 常数相同）。
        self.assertEqual(cyclic_mul([6], [7], 1).coefficients, [42])
        self.assertEqual(negacyclic_mul([6], [7], 1).coefficients, [42])
        self.assertEqual(cyclic_mul([-9], [-9], 1).coefficients, [81])
        # 输入长度不得超过 N=1。
        with self.assertRaises(LengthOutOfBoundError):
            cyclic_mul([1, 2], [3], 1)

    def test_zero_polynomial(self):
        # 零多项式结果是长度恰为 N 的全零（规范长度，不是空）。
        for n in (1, 8, 256):
            for negative in (False, True):
                result = _check_pair(self, [0] * n, [0] * n, n, negative)
                self.assertEqual(result.coefficients, [0] * n)
                self.assertEqual(result.coefficient_bound, 0)

    def test_input_shorter_than_n_is_padded_in_output_only(self):
        # 短输入不预约化：结果长度仍恰为 N。
        result = cyclic_mul([2], [3], 8)
        self.assertEqual(result.coefficients, [6] + [0] * 7)
        self.assertEqual(len(result.coefficients), 8)

    def test_large_coefficient_cancellation(self):
        # 大系数折返后大幅抵消：线性系数远超单模数（约 1e9），必须多模
        # CRT 先恢复有符号真值，再在整数上相减，不能在模 p 余数上变号。
        n = 256
        a = [MAX_COEFF_ABS] * n
        b = [MAX_COEFF_ABS] * n
        # 线性卷积 C 的峰在中部 256*1e12；折返到环后余数 r 收到两块。
        C = naive_convolve(a, b)
        cyc = cyclic_mul(a, b, n)
        neg = negacyclic_mul(a, b, n)
        self.assertEqual(cyc.coefficients, naive_ring_mul(a, b, n, False))
        self.assertEqual(neg.coefficients, naive_ring_mul(a, b, n, True))
        # 折返界必须是相加界（2 块），且模数乘积覆盖它。
        self.assertEqual(cyc.coefficient_bound, 2 * 256 * 10 ** 12)
        self.assertGreater(cyc.modulus_product, 2 * cyc.coefficient_bound)
        self.assertEqual(len(cyc.moduli), 2)
        # 负循环余数 0 收到 C[0] - C[256]：1e12 - 256e12 = -255e12，
        # 若在模 p 余数上错误变号/忘记有符号重建必得不到该值。
        self.assertEqual(neg.coefficients[0], C[0] - C[n])
        # C[256] 只含 i=1..255 共 255 个乘积项，即 255e12。
        self.assertEqual(C[n], 255 * 10 ** 12)
        self.assertEqual(neg.coefficients[0], -254 * 10 ** 12)
        self.assertGreater(abs(neg.coefficients[0]), 10 ** 9)
        # 余数 255 只收到一块（C[255]；k=511 不存在），两种环结果相同。
        self.assertEqual(cyc.coefficients[255], neg.coefficients[255])
        self.assertEqual(cyc.coefficients[255], C[255])
        self.assertEqual(cyc.coefficients[255], 256 * 10 ** 12)

    def test_deterministic_random_matrix(self):
        rng = random.Random(20260925)
        for n in (2, 8, 32, 128, 256):
            for negative in (False, True):
                la = 1 + rng.randrange(n)
                lb = 1 + rng.randrange(n)
                hi = 10 ** rng.randrange(1, 7)
                a = [rng.randrange(-hi, hi + 1) for _ in range(la)]
                b = [rng.randrange(-hi, hi + 1) for _ in range(lb)]
                _check_pair(self, a, b, n, negative)

    def test_matches_linear_convolution_when_no_fold(self):
        # n+m-1 <= N 时不发生折返，环结果就是线性卷积（后补零）。
        a = [3, -1, 4]
        b = [2, 5]
        for negative in (False, True):
            result = (negacyclic_mul if negative else cyclic_mul)(a, b, 16)
            self.assertEqual(result.coefficients[:4], convolve(a, b).coefficients)
            self.assertEqual(result.coefficients[4:], [0] * 12)


class RingValidationTests(unittest.TestCase):
    def test_length_exceeding_n_rejected(self):
        with self.assertRaises(LengthOutOfBoundError):
            cyclic_mul([1, 2, 3, 4, 5], [1], 4)
        with self.assertRaises(LengthOutOfBoundError):
            negacyclic_mul([1], [1, 2], 1)

    def test_bad_n(self):
        for bad_n in (0, -1, 3, 6, 512, 1.0, True, False, "8", None):
            with self.subTest(bad_n=bad_n):
                with self.assertRaises(ExactConvolutionError):
                    cyclic_mul([1], [1], bad_n)
                with self.assertRaises(ExactConvolutionError):
                    negacyclic_mul([1], [1], bad_n)

    def test_n_power_of_two_boundary_values(self):
        for n in (1, 2, 4, 256):
            self.assertEqual(cyclic_mul([1], [1], n).ring_size, n)
        for bad_n in (3, 5, 7, 100, 127, 255, 257):
            with self.assertRaises(ExactConvolutionError):
                cyclic_mul([1], [1], bad_n)

    def test_coefficient_rules_inherited(self):
        # bool/float/超系数界/空/非可迭代：与 convolve 完全一致。
        with self.assertRaises(InvalidCoefficientError):
            cyclic_mul([True, 2], [3], 4)
        with self.assertRaises(InvalidCoefficientError):
            negacyclic_mul([1.0], [2], 4)
        with self.assertRaises(InvalidCoefficientError):
            cyclic_mul([float("nan")], [2], 4)
        with self.assertRaises(CoefficientBoundError):
            negacyclic_mul([MAX_COEFF_ABS + 1], [1], 4)
        with self.assertRaises(LengthOutOfBoundError):
            cyclic_mul([], [1], 4)
        with self.assertRaises(InvalidPolynomialError):
            cyclic_mul(123, [1], 4)

    def test_inputs_not_mutated(self):
        a, b = [1, 2, 3], [4, 5, 6]
        sa, sb = list(a), list(b)
        cyclic_mul(a, b, 4)
        negacyclic_mul(a, b, 4)
        self.assertEqual((a, b), (sa, sb))

    def test_result_equality_and_repr(self):
        r1 = cyclic_mul([1, 2], [3, 4], 4)
        r2 = cyclic_mul([1, 2], [3, 4], 4)
        self.assertEqual(r1, r2)
        self.assertEqual(hash(r1), hash(r2))
        self.assertIn("x^4 - 1", repr(r1))
        r3 = negacyclic_mul([1, 2], [3, 4], 4)
        self.assertNotEqual(r1, r3)


if __name__ == "__main__":
    unittest.main()
