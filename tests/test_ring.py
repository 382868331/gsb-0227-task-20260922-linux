"""cyclic_mul / negacyclic_mul 端到端测试。

期望值全部来自 tests.reference 中的独立双循环折返实现（不导入被测核心
计算期望值）；NTT/CRT 相关性质仅作补充检验。
"""

import random
import unittest

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_RING_SIZE,
    RING_SIZES,
    InvalidCoefficientError,
    InvalidPolynomialError,
    InvalidRingSizeError,
    LengthOutOfBoundError,
    convolve,
    cyclic_mul,
    multiply,
    negacyclic_mul,
)

from tests.reference import (
    naive_convolve,
    naive_cyclic_mul,
    naive_negacyclic_mul,
)


def _fold_linear(linear, size, mode):
    """用已验证的线性卷积结果折返，仅作补充交叉检验。"""
    result = [0] * size
    for exponent, value in enumerate(linear):
        block, remainder = divmod(exponent, size)
        if mode == "negacyclic" and block & 1:
            result[remainder] -= value
        else:
            result[remainder] += value
    return result


def _assert_result_invariants(test, result, size, mode, a, b):
    """检查元数据不变量与折返界可证性。"""
    test.assertEqual(len(result.coefficients), size)
    test.assertEqual(result.ring_size, size)
    test.assertEqual(result.mode, mode)
    n, m = max(1, len(a)), max(1, len(b))
    amax = max((abs(x) for x in a), default=0)
    bmax = max((abs(x) for x in b), default=0)
    expected_folded_bound = min(size, n + m - 1) * amax * bmax
    expected_linear_bound = min(n, m) * amax * bmax
    test.assertEqual(result.coefficient_bound, expected_folded_bound)
    test.assertEqual(result.linear_coefficient_bound, expected_linear_bound)
    product = 1
    for mod in result.moduli:
        product *= mod
    test.assertEqual(result.modulus_product, product)
    test.assertGreater(product, 2 * expected_folded_bound)
    # 每个实际系数都不超过声明的统一界。
    for value in result.coefficients:
        test.assertLessEqual(abs(value), expected_folded_bound)
    test.assertTrue(result.transform_length & (result.transform_length - 1) == 0)
    test.assertGreaterEqual(result.transform_length, n + m - 1)


class CyclicCorrectnessTests(unittest.TestCase):
    def test_hand_picked_cyclic_wraparound(self):
        # 高次项折返到低次并相加：(1+2x)*(3+4x) = 3+10x+8x^2，
        # mod (x^2-1) 下 x^2 ≡ 1 => [3+8, 10] = [11, 10]。
        result = cyclic_mul([1, 2], [3, 4], 2)
        self.assertEqual(result.coefficients, [11, 10])
        self.assertEqual(result.coefficients,
                         naive_cyclic_mul([1, 2], [3, 4], 2))

    def test_highest_times_highest(self):
        # 最高次乘最高次：x^(N-1) * x^(N-1) = x^(2N-2) = x^(N-2)（循环）。
        for size in (1, 2, 4, 16, 256):
            a = [0] * size
            b = [0] * size
            a[-1] = 1
            b[-1] = 1
            expected = [0] * size
            expected[(size - 2) % size] = 1
            self.assertEqual(cyclic_mul(a, b, size).coefficients, expected)

    def test_short_inputs_and_room_for_more(self):
        # 输入长度远小于 N：无折返，结果就是补零到 N 的普通卷积。
        a, b, size = [1, -2, 3], [4, 5], 8
        result = cyclic_mul(a, b, size)
        expected = naive_convolve(a, b) + [0] * (size - len(a) - len(b) + 1)
        self.assertEqual(result.coefficients, expected)
        _assert_result_invariants(self, result, size, "cyclic", a, b)

    def test_deterministic_random_all_sizes(self):
        rng = random.Random(20260925)
        for size in RING_SIZES:
            n = rng.randrange(0, size + 1)
            m = rng.randrange(0, size + 1)
            a = [rng.randrange(-5000, 5001) for _ in range(n)]
            b = [rng.randrange(-5000, 5001) for _ in range(m)]
            result = cyclic_mul(a, b, size)
            self.assertEqual(result.coefficients,
                             naive_cyclic_mul(a, b, size))
            _assert_result_invariants(self, result, size, "cyclic", a, b)

    def test_max_size_max_magnitude(self):
        a = [MAX_COEFF_ABS] * MAX_RING_SIZE
        b = [-MAX_COEFF_ABS] * MAX_RING_SIZE
        result = cyclic_mul(a, b, MAX_RING_SIZE)
        self.assertEqual(result.coefficients,
                         naive_cyclic_mul(a, b, MAX_RING_SIZE))
        _assert_result_invariants(self, result, MAX_RING_SIZE, "cyclic", a, b)

    def test_inputs_are_not_mutated(self):
        a = [1, 2, 3, 0]
        b = [-1, 0, 2, 5]
        a_copy, b_copy = list(a), list(b)
        cyclic_mul(a, b, 4)
        negacyclic_mul((x for x in a), (x for x in b), 4)
        self.assertEqual(a, a_copy)
        self.assertEqual(b, b_copy)


class NegacyclicCorrectnessTests(unittest.TestCase):
    def test_hand_picked_sign_difference(self):
        # 同样输入与 N=2：3+10x+8x^2，mod (x^2+1) 下 x^2 ≡ -1
        # => [3-8, 10] = [-5, 10]；与循环结果 [11, 10] 的常数项差一个符号。
        a, b = [1, 2], [3, 4]
        self.assertEqual(negacyclic_mul(a, b, 2).coefficients, [-5, 10])
        self.assertEqual(cyclic_mul(a, b, 2).coefficients, [11, 10])
        self.assertEqual(negacyclic_mul(a, b, 2).coefficients,
                         naive_negacyclic_mul(a, b, 2))

    def test_highest_times_highest(self):
        # x^(N-1) * x^(N-1) = x^(2N-2) = -x^(N-2)（负循环），
        # N=2 时 N-2=0：即 -1；N=1 时退化为常数乘积带块奇偶。
        for size in (2, 4, 16, 256):
            a = [0] * size
            b = [0] * size
            a[-1] = 1
            b[-1] = 1
            expected = [0] * size
            expected[size - 2] = -1
            self.assertEqual(negacyclic_mul(a, b, size).coefficients, expected)

    def test_block_sign_difference(self):
        # N=2，合法输入最长 2，线性卷积最长 3，恰好有一个跨块项：
        # (1+x)^2 = 1 + 2x + x^2。循环 x^2 ≡ 1 => [2, 2]；
        # 负循环 x^2 ≡ -1 => [0, 2]。
        a = b = [1, 1]
        self.assertEqual(cyclic_mul(a, b, 2).coefficients, [2, 2])
        self.assertEqual(negacyclic_mul(a, b, 2).coefficients, [0, 2])
        # 更长的跨两块需要输入长度不超过 N，取 N=4、长度 4：
        # (1+x+x^2+x^3)^2 的指数 4 项（3 个乘积对）折返到常数项，
        # 循环加、负循环减，差值是该项的两倍。
        c = [1, 1, 1, 1]
        cyc = cyclic_mul(c, c, 4).coefficients
        neg = negacyclic_mul(c, c, 4).coefficients
        self.assertEqual(cyc, naive_cyclic_mul(c, c, 4))
        self.assertEqual(neg, naive_negacyclic_mul(c, c, 4))
        self.assertEqual(cyc[0] - neg[0], 6)

    def test_large_coefficient_cancellation(self):
        # 大系数在折返中部分相消：线性系数远超单模数，折返后常数项抵消，
        # 验证 CRT 先重建有符号整数、再施加负号，而不是模 p 取负。
        size = 16
        a = [0] * size
        b = [0] * size
        a[0] = MAX_COEFF_ABS
        a[size - 1] = MAX_COEFF_ABS
        b[0] = MAX_COEFF_ABS
        b[1] = MAX_COEFF_ABS
        # 线性积含项 +1e12（指数 0）与 +1e12（来自 (N-1)+1=N，折返到 0），
        # 负循环下指数 N 带负号：常数项 1e12 - 1e12 = 0；同时指数 1 和
        # N-1 各有一个 +1e12，不参与常数项。
        result = negacyclic_mul(a, b, size)
        self.assertEqual(result.coefficients[0], 0)
        self.assertEqual(result.coefficients,
                         naive_negacyclic_mul(a, b, size))
        self.assertGreaterEqual(len(result.moduli), 2)
        _assert_result_invariants(self, result, size, "negacyclic", a, b)
        # 同一输入循环版本常数项为 2e12（同号相加），符号差异可见。
        cyc = cyclic_mul(a, b, size)
        self.assertEqual(cyc.coefficients[0], 2 * MAX_COEFF_ABS ** 2)

    def test_deterministic_random_all_sizes(self):
        rng = random.Random(998244353)
        for size in RING_SIZES:
            n = rng.randrange(0, size + 1)
            m = rng.randrange(0, size + 1)
            a = [rng.choice((-1, 1)) * rng.randrange(0, 3000)
                 for _ in range(n)]
            b = [rng.choice((-1, 1)) * rng.randrange(0, 3000)
                 for _ in range(m)]
            result = negacyclic_mul(a, b, size)
            self.assertEqual(result.coefficients,
                             naive_negacyclic_mul(a, b, size))
            _assert_result_invariants(self, result, size, "negacyclic",
                                      a, b)
            # 补充：与“精确线性卷积再折返”完全一致。
            linear = convolve(a if a else [0], b if b else [0]).coefficients
            self.assertEqual(result.coefficients,
                             _fold_linear(linear, size, "negacyclic"))

    def test_max_size_max_magnitude(self):
        a = [MAX_COEFF_ABS] * MAX_RING_SIZE
        b = [MAX_COEFF_ABS] * MAX_RING_SIZE
        result = negacyclic_mul(a, b, MAX_RING_SIZE)
        self.assertEqual(result.coefficients,
                         naive_negacyclic_mul(a, b, MAX_RING_SIZE))
        _assert_result_invariants(self, result, MAX_RING_SIZE,
                                  "negacyclic", a, b)


class ZeroPolynomialAndNOneTests(unittest.TestCase):
    def test_zero_polynomial_variants(self):
        for size in (1, 2, 8, 256):
            zero = [0] * size
            for za, zb in (([], []), ([], [1]), (zero, [1]),
                           ([0, 0] if size >= 2 else [0], [])):
                self.assertEqual(cyclic_mul(za, zb, size).coefficients,
                                 [0] * size)
                self.assertEqual(negacyclic_mul(za, zb, size).coefficients,
                                 [0] * size)

    def test_n_one_constant_ring(self):
        # N=1 两个商环都是常数环：cyclic 下 x ≡ 1，negacyclic 下 x ≡ -1，
        # 但常数多项式相乘两者相同。
        self.assertEqual(cyclic_mul([7], [-3], 1).coefficients, [-21])
        self.assertEqual(negacyclic_mul([7], [-3], 1).coefficients, [-21])
        self.assertEqual(cyclic_mul([], [5], 1).coefficients, [0])
        self.assertEqual(negacyclic_mul([0], [], 1).coefficients, [0])


class RingValidationTests(unittest.TestCase):
    def test_bad_ring_sizes_rejected(self):
        for bad in (0, 3, 5, 6, 7, 100, 257, 512, -4, 1.0, "4", None,
                    True, False):
            with self.assertRaises(InvalidRingSizeError):
                cyclic_mul([1], [1], bad)
            with self.assertRaises(InvalidRingSizeError):
                negacyclic_mul([1], [1], bad)

    def test_input_longer_than_N_rejected_not_truncated(self):
        for func in (cyclic_mul, negacyclic_mul):
            with self.assertRaises(LengthOutOfBoundError):
                func([1, 2, 3], [1], 2)
            with self.assertRaises(LengthOutOfBoundError):
                func([1], [0, 0, 0, 0, 0], 4)
            # 长度恰好 N 合法。
            result = func([1, 2], [3, 4], 2)
            self.assertEqual(len(result.coefficients), 2)

    def test_coefficient_validation_shared_with_convolve(self):
        for func in (cyclic_mul, negacyclic_mul):
            with self.assertRaises(InvalidCoefficientError):
                func([1, True], [1], 4)
            with self.assertRaises(InvalidCoefficientError):
                func([1, 2.0], [1], 4)
            with self.assertRaises(InvalidPolynomialError):
                func(123, [1], 4)

    def test_multiply_alias_preserves_linear_interface(self):
        self.assertIs(multiply, convolve)
        result = multiply([1, 2, 3], [4, 5])
        self.assertEqual(result.coefficients, [4, 13, 22, 15])


if __name__ == "__main__":
    unittest.main()
