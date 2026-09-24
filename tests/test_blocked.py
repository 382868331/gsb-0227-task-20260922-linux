"""固定块长重叠累加卷积的测试。

期望值来源（均不调用被测分块实现生成）：
- 手算硬编码的小例子；
- ``tests.reference.naive_convolve``（独立朴素双重循环，不导入 exact_conv）；
- 小规模*枚举*：{-1,0,1} 上所有短序列对、多种块长逐一核对；
- 解析关系：块记录的偏移必须等于 (i+j)*S，输出长度 n+m-1，块对总数
  ceil(n/S)*ceil(m/S)，统计字段由块长按定义独立重算。
"""

import itertools
import random
import unittest

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    CoefficientBoundError,
    InvalidBlockSizeError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    blocked_convolve,
    convolve,
)

from tests.reference import naive_convolve


def _ceil_div(x, y):
    return (x + y - 1) // y


def _check_stats(test, result, n, m, block_size):
    """用块长定义独立核对全部统计字段。"""
    stats = result.working_set
    na = _ceil_div(n, block_size)
    nb = _ceil_div(m, block_size)
    test.assertEqual(stats.block_size, block_size)
    test.assertEqual(stats.blocks_a, na)
    test.assertEqual(stats.blocks_b, nb)
    # 零块不跳过：块对一次不少。
    test.assertEqual(stats.block_calls, na * nb)
    test.assertEqual(len(result.block_calls), na * nb)
    test.assertEqual(stats.output_slots, n + m - 1)
    test.assertEqual(len(result.coefficients), n + m - 1)

    expected_slots = []
    for idx, rec in enumerate(result.block_calls):
        la = min(block_size, n - rec.block_a * block_size)
        lb = min(block_size, m - rec.block_b * block_size)
        test.assertEqual(rec.len_a, la)
        test.assertEqual(rec.len_b, lb)
        # 解析关系：块卷积起始偏移 = (i+j)*S，调用顺序按 i 外层 j 内层。
        test.assertEqual(rec.block_a, idx // nb)
        test.assertEqual(rec.block_b, idx % nb)
        test.assertEqual(rec.offset, (rec.block_a + rec.block_b) * block_size)
        tlen = rec.transform_length
        test.assertTrue(tlen & (tlen - 1) == 0)
        test.assertGreaterEqual(tlen, la + lb - 1)
        expected_slots.append(3 * tlen)
    test.assertEqual(list(stats.per_call_ntt_slots), expected_slots)
    test.assertEqual(list(stats.transform_lengths),
                     [r.transform_length for r in result.block_calls])
    test.assertEqual(stats.peak_ntt_slots, max(expected_slots))
    test.assertEqual(stats.total_ntt_slots, sum(expected_slots))
    test.assertEqual(stats.peak_working_slots,
                     stats.output_slots + stats.peak_ntt_slots)


class BlockedCorrectnessTests(unittest.TestCase):
    def test_hand_picked_with_negative_coefficients(self):
        # 手算：(1 - 2x + 3x^2)(4 - x) = 4 - 9x + 14x^2 - 3x^3
        a = [1, -2, 3]
        b = [4, -1]
        hardcoded = [4, -9, 14, -3]
        for block_size in (1, 2, 3, 7):
            with self.subTest(block_size=block_size):
                result = blocked_convolve(a, b, block_size)
                self.assertEqual(result.coefficients, hardcoded)
                self.assertEqual(result.coefficients,
                                 naive_convolve(a, b))

    def test_exhaustive_small_sequences_all_block_sizes(self):
        # 小规模枚举：长度 1..3 的所有 {-1,0,1} 序列对，块长 1/2/3/5，
        # 共 (3+9+27)^2 * 4 = 6084 次真实卷积，全部对拍朴素参考。
        sequences = {}
        for length in range(1, 4):
            sequences[length] = [list(seq)
                                 for seq in itertools.product((-1, 0, 1),
                                                              repeat=length)]
        for n, m in itertools.product(range(1, 4), repeat=2):
            expected_pairs = [(a, b, naive_convolve(a, b))
                              for a in sequences[n] for b in sequences[m]]
            for block_size in (1, 2, 3, 5):
                for a, b, expected in expected_pairs:
                    result = blocked_convolve(a, b, block_size)
                    self.assertEqual(result.coefficients, expected)

    def test_block_boundaries_exact_multiple_and_plus_one(self):
        # 长度恰为块长整数倍、倍数 +1、倍数 -1 三种边界。
        rng = random.Random(20260924)
        for n, m, block_size in [
                (6, 6, 3),     # 恰整数倍：2x2 块
                (7, 4, 3),     # n = 2S+1（末块长 1），m = S+1
                (5, 8, 3),     # n = 2S-1（末块长 2）
                (1, 1, 1),
                (4, 1, 256),   # 块长远大于输入：每侧只 1 块
        ]:
            with self.subTest(n=n, m=m, block_size=block_size):
                a = [rng.randrange(-50, 51) for _ in range(n)]
                b = [rng.randrange(-50, 51) for _ in range(m)]
                result = blocked_convolve(a, b, block_size)
                self.assertEqual(result.coefficients, naive_convolve(a, b))
                _check_stats(self, result, n, m, block_size)

    def test_block_records_match_legacy_convolution_of_slices(self):
        # 每个块对的结果必须等于旧入口对该切片的精确卷积，且按偏移拼回。
        a = [3, -7, 0, 2, 5, -1, 4]
        b = [1, 2, -3, 0, 6]
        block_size = 3
        result = blocked_convolve(a, b, block_size)
        for rec in result.block_calls:
            sa = a[rec.block_a * block_size:
                   (rec.block_a + 1) * block_size]
            sb = b[rec.block_b * block_size:
                   (rec.block_b + 1) * block_size]
            legacy = convolve(sa, sb)
            self.assertEqual(rec.moduli, legacy.moduli)
            self.assertEqual(rec.transform_length, legacy.transform_length)

    def test_zero_blocks_are_still_processed(self):
        # a 切成 [1,0] [0,0] [0,1]：中间是零块，必须照常调用旧卷积
        # （块对不跳过），且结果仍精确。
        a = [1, 0, 0, 0, 0, 1]
        b = [2, -1, 3]
        block_size = 2
        result = blocked_convolve(a, b, block_size)
        self.assertEqual(result.coefficients, naive_convolve(a, b))
        # 3x2 = 6 次调用一次不少；涉及零块的记录变换长度仍按形状给出。
        self.assertEqual(result.working_set.block_calls, 6)
        zero_block_records = [r for r in result.block_calls
                              if r.block_a == 1]
        self.assertEqual(len(zero_block_records), 2)
        for rec in zero_block_records:
            self.assertEqual(rec.len_a, 2)
            # 独立重算该零块与对应 B 块的卷积：必须逐系数为零，
            # 证明“零块也调用旧卷积”而不是被静默跳过后编造记录。
            sb = b[rec.block_b * block_size:
                   (rec.block_b + 1) * block_size]
            self.assertEqual(convolve([0, 0], sb).coefficients,
                             [0] * (rec.len_a + rec.len_b - 1))

    def test_all_zero_and_zero_tail_inputs(self):
        self.assertEqual(blocked_convolve([0, 0, 0, 0], [1, -2, 3], 2).
                         coefficients, [0] * 6)
        a = [0, 1, 0] + [0] * 10
        b = [0, 0, -1, 2] + [0] * 10
        result = blocked_convolve(a, b, 4)
        self.assertEqual(result.coefficients, naive_convolve(a, b))
        self.assertTrue(all(x == 0 for x in result.coefficients[:2]))

    def test_inputs_longer_than_legacy_limit(self):
        # 新能力：长度超过旧入口 256 上限。
        rng = random.Random(777)
        for n, m, block_size in [(257, 257, 64), (300, 1, 128),
                                 (301, 299, 100), (1000, 3, 256)]:
            with self.subTest(n=n, m=m, block_size=block_size):
                a = [rng.randrange(-1000, 1001) for _ in range(n)]
                b = [rng.randrange(-1000, 1001) for _ in range(m)]
                result = blocked_convolve(a, b, block_size)
                self.assertEqual(result.coefficients, naive_convolve(a, b))
                _check_stats(self, result, n, m, block_size)

    def test_large_coefficients_overflow_block_modulus_but_exact(self):
        # 块内系数 1e6、块长 256：块卷积峰值 256e12 必须走双模 CRT；
        # 跨块重叠累加后，总输出峰值 384e12（>任何单块界），仍然精确。
        n = 384
        a = [MAX_COEFF_ABS] * n
        b = [-MAX_COEFF_ABS] * n
        result = blocked_convolve(a, b, 256)
        self.assertEqual(result.coefficients, naive_convolve(a, b))
        # 所有满块记录都使用两个模数。
        full_records = [r for r in result.block_calls
                        if r.len_a == 256 and r.len_b == 256]
        self.assertTrue(full_records)
        self.assertTrue(all(len(r.moduli) == 2 for r in full_records))
        peak = min(n, n) * MAX_COEFF_ABS * MAX_COEFF_ABS
        self.assertEqual(max(abs(x) for x in result.coefficients), peak)
        # 峰值远超任一模数：证明不是取模冒充。
        self.assertGreater(peak, 10 ** 11)

    def test_negative_times_positive_sign_structure(self):
        a = [-1, -2, -3, -4, -5, -6]
        b = [1, 2, 3, 4]
        for block_size in (1, 2, 3, 4, 5):
            result = blocked_convolve(a, b, block_size)
            self.assertEqual(result.coefficients, naive_convolve(a, b))
            self.assertTrue(all(x < 0 for x in result.coefficients))

    def test_result_independent_of_block_size(self):
        rng = random.Random(31337)
        a = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(50)]
        b = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(37)]
        outputs = {tuple(blocked_convolve(a, b, s).coefficients)
                   for s in (1, 7, 16, 32, 64, 256)}
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs.pop(), tuple(naive_convolve(a, b)))

    def test_inputs_not_mutated_and_generators_accepted(self):
        a, b = [1, -2, 0, 4, 5], [2, 3, -1]
        sa, sb = list(a), list(b)
        blocked_convolve(a, b, 2)
        self.assertEqual((a, b), (sa, sb))
        result = blocked_convolve(iter(a), tuple(b), 3)
        self.assertEqual(result.coefficients, naive_convolve(a, b))

    def test_legacy_entry_unchanged_still_rejects_long_input(self):
        # 旧契约保持：convolve 依旧拒绝超过 256 的输入。
        with self.assertRaises(LengthOutOfBoundError):
            convolve([1] * (MAX_POLY_LENGTH + 1), [1])


class BlockedValidationTests(unittest.TestCase):
    def test_bad_block_size(self):
        a, b = [1, 2], [3, 4]
        for bad in (0, -1, -256, MAX_POLY_LENGTH + 1, 10 ** 9):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidBlockSizeError):
                    blocked_convolve(a, b, bad)

    def test_block_size_wrong_type(self):
        a, b = [1, 2], [3, 4]
        for bad in (True, False, 1.0, "3", None, 3.0):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidBlockSizeError):
                    blocked_convolve(a, b, bad)

    def test_bad_polynomial_inputs(self):
        with self.assertRaises(InvalidPolynomialError):
            blocked_convolve(123, [1, 2], 2)
        with self.assertRaises(InvalidPolynomialError):
            blocked_convolve([1, 2], "xy", 2)
        with self.assertRaises(LengthOutOfBoundError):
            blocked_convolve([], [1], 2)
        with self.assertRaises(InvalidCoefficientError):
            blocked_convolve([True, 2], [1], 2)
        with self.assertRaises(InvalidCoefficientError):
            blocked_convolve([1], [float("nan")], 2)
        with self.assertRaises(InvalidCoefficientError):
            blocked_convolve([1.5], [2], 2)

    def test_coefficient_bound_error_with_global_index(self):
        # 超长输入中靠后位置（第 300 项，超过旧 256 上限）的超界系数：
        # 必须拒绝，错误下标按原序列全局位置给出。
        a = [1] * 301
        a[300] = MAX_COEFF_ABS + 1
        with self.assertRaises(CoefficientBoundError) as ctx:
            blocked_convolve(a, [1, 2], 64)
        self.assertIn("a[300]", str(ctx.exception))

    def test_boundary_coefficient_accepted(self):
        result = blocked_convolve([MAX_COEFF_ABS, -MAX_COEFF_ABS],
                                  [MAX_COEFF_ABS], 1)
        self.assertEqual(result.coefficients, [10 ** 12, -(10 ** 12)])

    def test_errors_are_value_errors_and_locatable(self):
        for bad_bs in (0, True, 2.0):
            with self.assertRaises(ValueError):
                blocked_convolve([1], [1], bad_bs)
        with self.assertRaises(InvalidCoefficientError) as ctx:
            blocked_convolve([1, 2, 3, 4, "bad"], [1], 2)
        self.assertIn("a[4]", str(ctx.exception))

    def test_no_partial_state_on_failure(self):
        # 校验在累加器建立之前完成：失败调用不改动输入，也不返回半成品。
        a = [1, 2, 3, 4]
        b = [1, 10 ** 6 + 1]
        snapshot_a, snapshot_b = list(a), list(b)
        with self.assertRaises(CoefficientBoundError):
            blocked_convolve(a, b, 2)
        self.assertEqual((a, b), (snapshot_a, snapshot_b))
        with self.assertRaises(InvalidBlockSizeError):
            blocked_convolve(a, b, 0)
        self.assertEqual((a, b), (snapshot_a, snapshot_b))


if __name__ == "__main__":
    unittest.main()
