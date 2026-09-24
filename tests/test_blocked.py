"""convolve_blocked 固定块长入口的测试。

期望值（卷积系数）全部来自独立朴素参考 tests.reference.naive_convolve，
不使用被测实现生成。工作数组统计等元数据用解析公式/手算值独立复核。
"""

import itertools
import random
import unittest

from exact_conv import (
    MAX_COEFF_ABS,
    MAX_POLY_LENGTH,
    MODULI,
    BlockedConvolutionResult,
    CoefficientBoundError,
    ExactConvolutionError,
    InvalidBlockLengthError,
    InvalidCoefficientError,
    InvalidPolynomialError,
    LengthOutOfBoundError,
    WorkArrayStats,
    convolve,
    convolve_blocked,
)

from tests.reference import naive_convolve


def _pow2_ge(value):
    """测试侧独立的“>= value 的最小 2 的幂”，不调用被测实现。"""
    return 1 << (value - 1).bit_length()


def _expected_mod_count(max_abs_pair_product, pair_short_len):
    """按题面规则在测试侧独立推算一对块需要的模数个数。

    B = min(块长a, 块长b) * max|A块| * max|B块|，取固定模数前缀中第一个
    使乘积 > 2B 的前缀。
    """
    bound = pair_short_len * max_abs_pair_product
    product = 1
    count = 0
    for mod in MODULI:
        product *= mod
        count += 1
        if product > 2 * bound:
            return count
    raise AssertionError("fixed moduli insufficient for test case")


def _assert_blocked_matches_naive(test, a, b, block_size):
    """分块结果逐系数等于独立朴素参考，并校验结构与统计的自洽关系。"""
    result = convolve_blocked(a, b, block_size)
    expected = naive_convolve(a, b)
    test.assertIsInstance(result, BlockedConvolutionResult)
    test.assertEqual(result.coefficients, expected)
    test.assertEqual(len(result.coefficients), len(a) + len(b) - 1)
    test.assertEqual(result.block_size, block_size)

    n, m = len(a), len(b)
    ba = (n + block_size - 1) // block_size
    bb = (m + block_size - 1) // block_size
    test.assertEqual((result.blocks_a, result.blocks_b), (ba, bb))
    # 零块也照常调用：块对调用数恒为块数之积。
    test.assertEqual(result.block_pair_calls, ba * bb)

    # ---- 工作数组统计：全部用输入长度独立重算 ----
    work = result.work
    test.assertIsInstance(work, WorkArrayStats)
    test.assertEqual(work.accumulator_slots, n + m - 1)

    exp_ntt_total = 0
    exp_result_total = 0
    exp_max_rows = 0
    pair_lens = []
    max_ntt = 0
    for i in range(ba):
        ai = a[i * block_size:(i + 1) * block_size]
        for j in range(bb):
            bj = b[j * block_size:(j + 1) * block_size]
            pair_len = len(ai) + len(bj) - 1
            pair_lens.append(pair_len)
            ntt_len = _pow2_ge(pair_len)
            max_ntt = max(max_ntt, ntt_len)
            exp_ntt_total += ntt_len
            exp_result_total += pair_len
            need = _expected_mod_count(
                max(map(abs, ai)) * max(map(abs, bj)),
                min(len(ai), len(bj)))
            exp_max_rows = max(exp_max_rows, need * pair_len)
    test.assertEqual(work.transform_length, max_ntt)
    test.assertEqual(work.total_ntt_slots, exp_ntt_total)
    test.assertEqual(work.total_block_result_slots, exp_result_total)
    test.assertEqual(work.max_residue_row_slots, exp_max_rows)
    # 峰值上界必须真的是上界：覆盖累加数组 + 单次块卷积的同存数组。
    test.assertGreaterEqual(
        work.peak_working_slots,
        work.accumulator_slots + 3 * max_ntt + exp_max_rows)
    test.assertGreaterEqual(
        work.peak_working_slots, work.accumulator_slots + max(pair_lens))
    return result


class HandPickedBlockedTests(unittest.TestCase):
    def test_hand_computed_example_odd_block_boundary(self):
        # 手算：(1+2x+3x^2+4x^3+5x^4)(1-x+2x^2)
        #   = 1 + x + 3x^2 + 5x^3 + 7x^4 + 3x^5 + 10x^6
        a = [1, 2, 3, 4, 5]
        b = [1, -1, 2]
        # 块长 2：a 切成 [1,2|[3,4|[5]]，b 切成 [1,-1|[2]]，
        # 首尾都是不完整块，专门压块边界与错位累加。
        result = convolve_blocked(a, b, 2)
        self.assertEqual(result.coefficients, [1, 1, 3, 5, 7, 3, 10])
        self.assertEqual(result.blocks_a, 3)
        self.assertEqual(result.blocks_b, 2)
        self.assertEqual(result.block_pair_calls, 6)

    def test_hand_example_all_block_sizes(self):
        # 同一手算例对每个合法块长都应一致（1 -> 整块；5 -> 单块）。
        a = [1, 2, 3, 4, 5]
        b = [1, -1, 2]
        expect = [1, 1, 3, 5, 7, 3, 10]
        for bs in range(1, 6):
            with self.subTest(block_size=bs):
                self.assertEqual(
                    convolve_blocked(a, b, bs).coefficients, expect)

    def test_negative_coefficients_mixed_signs(self):
        # 负系数：块卷积由 CRT 对称区间恢复，重叠累加为精确大整数加减。
        a = [-3, 2, -1, 4, -5, 0, 2]
        b = [1, -2, -3, 6]
        for bs in (1, 2, 3, 4, 7):
            with self.subTest(block_size=bs):
                _assert_blocked_matches_naive(self, a, b, bs)


class BlockBoundaryTests(unittest.TestCase):
    def test_lengths_exact_multiples_and_plus_one(self):
        # 长度 3..9，覆盖块长 1/3/4/8 下最后一块满、缺 1、缺多格等情形。
        for n in range(3, 10):
            for m in range(3, 10):
                a = [((i + 1) * (1 if i % 2 else -1)) for i in range(n)]
                b = [(7 - j) for j in range(m)]
                for bs in (1, 3, 4, 8):
                    with self.subTest(n=n, m=m, bs=bs):
                        _assert_blocked_matches_naive(self, a, b, bs)

    def test_block_size_one_is_schoolbook_positionwise(self):
        # 块长 1：每块一个系数，块对即 a_i*b_j，偏移 i+j —— 与学校乘法
        # 的每一项一一对应。
        a = [2, -3, 5, -7]
        b = [4, -1, -2]
        result = convolve_blocked(a, b, 1)
        self.assertEqual(result.coefficients, naive_convolve(a, b))
        self.assertEqual(result.block_pair_calls, len(a) * len(b))
        self.assertEqual(result.work.total_block_result_slots,
                         len(a) * len(b))
        # 1x1 块卷积长度恒为 1，NTT 长度恒为 1。
        self.assertEqual(result.work.transform_length, 1)
        self.assertEqual(result.work.total_ntt_slots, len(a) * len(b))

    def test_single_block_equals_old_convolve(self):
        # 输入不超过块长时，分块入口只有一对块，且结果与旧入口完全一致。
        rng = random.Random(20260924)
        for n, m in ((1, 1), (7, 3), (64, 64), (256, 200)):
            a = [rng.randrange(-10 ** 5, 10 ** 5 + 1) for _ in range(n)]
            b = [rng.randrange(-10 ** 5, 10 ** 5 + 1) for _ in range(m)]
            blocked = convolve_blocked(a, b, 256)
            direct = convolve(a, b)
            self.assertEqual(blocked.coefficients, direct.coefficients)
            self.assertEqual(blocked.block_pair_calls, 1)
            self.assertEqual(blocked.transform_length, direct.transform_length)


class ZeroBlockTests(unittest.TestCase):
    def test_interior_zero_blocks_still_counted(self):
        # 长度 9、块长 3：中间整块 [0,0,0]，两边非零；零块不跳过，
        # 块对调用数仍是 3*3=9，但零块参与时模数可降到单模。
        a = [1, 2, 3, 0, 0, 0, -4, 5, -6]
        b = [2, -1, 0, 0, 0, 0, 3, 3, -3]
        result = _assert_blocked_matches_naive(self, a, b, 3)
        self.assertEqual(result.block_pair_calls, 9)
        # 含非零块的对仍可能用到双模；零块对 B=0 只需要单模。
        self.assertEqual(result.moduli[0], MODULI[0])

    def test_all_zero_long_polynomials(self):
        # 远超 256 的全零输入：每个系数都必须是精确的 0。
        a = [0] * 600
        b = [0] * 401
        result = convolve_blocked(a, b, 64)
        self.assertEqual(result.coefficients, [0] * (600 + 401 - 1))
        self.assertEqual(result.block_pair_calls, 10 * 7)
        # 界 B=0：所有块对都只需单模。
        self.assertEqual(result.moduli, (MODULI[0],))
        self.assertEqual(result.work.max_residue_row_slots,
                         2 * 64 - 1)  # 最长块对 64+64-1

    def test_zero_multiplied_by_long_nonzero(self):
        a = [0] * 300
        rng = random.Random(77)
        b = [rng.randrange(-10 ** 6, 10 ** 6 + 1) for _ in range(257)]
        result = _assert_blocked_matches_naive(self, a, b, 32)
        self.assertTrue(all(x == 0 for x in result.coefficients))


class LargeCoefficientTests(unittest.TestCase):
    def test_max_magnitude_longer_than_old_limit(self):
        # 长度 300 > 旧入口上限 256；系数全取 ±10^6。中央系数
        # C[299] = Σ a_i b_{299-i}（300 项 ±1e12），朴素参考逐位核对。
        rng = random.Random(314159)
        a = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(300)]
        b = [rng.choice((-1, 1)) * MAX_COEFF_ABS for _ in range(300)]
        result = _assert_blocked_matches_naive(self, a, b, 64)
        self.assertEqual(result.blocks_a, 5)  # 64*4 + 44
        self.assertEqual(result.blocks_b, 5)
        self.assertGreater(max(map(abs, result.coefficients)), 10 ** 13)

    def test_all_positive_max_magnitude_peak_exact(self):
        # 全 +1e6，长度 300，块长 256（2x2 块对，最后一块 44 项）：
        # 峰值系数 = 300 * 10^12，精确可手算。
        a = [MAX_COEFF_ABS] * 300
        b = [MAX_COEFF_ABS] * 300
        result = convolve_blocked(a, b, 256)
        self.assertEqual(max(result.coefficients), 300 * 10 ** 12)
        self.assertEqual(result.coefficients[0], 10 ** 12)
        self.assertEqual(result.coefficients[-1], 10 ** 12)
        self.assertEqual(len(result.coefficients), 599)
        self.assertEqual(result.block_pair_calls, 4)

    def test_old_entry_rejects_long_input_new_entry_accepts(self):
        # 新能力确实解除了长度限制：旧入口对 257 长度拒绝，新入口正确。
        a = [1] * 257
        b = [1, -1]
        with self.assertRaises(LengthOutOfBoundError):
            convolve(a, b)
        result = convolve_blocked(a, b, 16)
        self.assertEqual(result.coefficients, naive_convolve(a, b))


class RandomMatrixTests(unittest.TestCase):
    def test_deterministic_random_sizes_and_block_sizes(self):
        rng = random.Random(20260924)
        cases = [
            (257, 1, 1, 10 ** 6),
            (257, 257, 256, 10 ** 6),
            (300, 199, 64, 10 ** 6),
            (513, 333, 100, 123456),
            (700, 700, 128, 10 ** 6),
            (1000, 37, 37, 10 ** 5),
            (1001, 1001, 200, 999999),
        ]
        for n, m, bs, hi in cases:
            with self.subTest(n=n, m=m, bs=bs, hi=hi):
                a = [rng.randrange(-hi, hi + 1) for _ in range(n)]
                b = [rng.randrange(-hi, hi + 1) for _ in range(m)]
                _assert_blocked_matches_naive(self, a, b, bs)

    def test_small_enumeration_all_sign_patterns(self):
        # 小规模完全枚举：长度 1..3，系数取自 {-1,0,1}，块长 1..3。
        for n, m in itertools.product(range(1, 4), repeat=2):
            for a in itertools.product((-1, 0, 1), repeat=n):
                for b in itertools.product((-1, 0, 1), repeat=m):
                    for bs in (1, 2, 3):
                        got = convolve_blocked(list(a), list(b), bs)
                        self.assertEqual(
                            got.coefficients, naive_convolve(list(a), list(b)))


class InputValidationTests(unittest.TestCase):
    def test_bad_block_size_rejected(self):
        a, b = [1, 2, 3, 4], [5, 6]
        for bad in (0, -1, MAX_POLY_LENGTH + 1, 10 ** 9,
                    1.5, True, False, None, "8", 2.0, [8]):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidBlockLengthError):
                    convolve_blocked(a, b, bad)

    def test_block_size_error_is_value_error(self):
        self.assertTrue(issubclass(InvalidBlockLengthError, ValueError))
        with self.assertRaises(ValueError):
            convolve_blocked([1], [1], 0)

    def test_long_input_still_validates_coefficient_types(self):
        # 长序列中的非法系数类型必须被拒绝（bool/float/NaN/字符串/None）。
        good_b = [1, 2]
        for bad_coeffs, etype in [
                ([1] * 300 + [True], InvalidCoefficientError),
                ([1] * 280 + [1.0], InvalidCoefficientError),
                ([float("nan")] + [1] * 300, InvalidCoefficientError),
                ([1] * 257 + ["x"], InvalidCoefficientError),
                ([1] * 257 + [None], InvalidCoefficientError),
        ]:
            with self.subTest(etype=etype):
                with self.assertRaises(etype):
                    convolve_blocked(bad_coeffs, good_b, 32)

    def test_long_input_bound_error_is_locatable(self):
        # 超界系数在第 300 位：错误信息必须可定位，且明确拒绝而非取模。
        a = [1] * 299 + [MAX_COEFF_ABS + 1]
        with self.assertRaises(CoefficientBoundError) as ctx:
            convolve_blocked(a, [2], 64)
        self.assertIn("a[299]", str(ctx.exception))

    def test_non_iterable_empty_and_string_rejected(self):
        with self.assertRaises(InvalidPolynomialError):
            convolve_blocked(123, [1, 2], 4)
        with self.assertRaises(InvalidPolynomialError):
            convolve_blocked([1, 2] * 200, None, 4)
        with self.assertRaises(InvalidPolynomialError):
            convolve_blocked("abcdef", [1, 2], 2)
        with self.assertRaises(LengthOutOfBoundError):
            convolve_blocked([], [1, 2], 4)
        with self.assertRaises(LengthOutOfBoundError):
            convolve_blocked([1] * 10, [], 4)

    def test_validation_before_allocation_no_partial_state(self):
        # 任何非法组合都不得返回半更新结果；函数无状态，重复调用互不影响。
        good = convolve_blocked([1, 2, 3, 4], [1, 1], 2)
        self.assertEqual(good.coefficients, [1, 3, 5, 7, 4])
        for bad_kwargs in (
                dict(a=[1, 2], b=[3], block_size=0),
                dict(a=[1, 2], b=[10 ** 6 + 1] * 300, block_size=4),
                dict(a=[], b=[1], block_size=4),
        ):
            with self.assertRaises(ExactConvolutionError):
                convolve_blocked(**bad_kwargs)
        # 失败后再调用，结果与首次完全一致，不存在残留状态。
        again = convolve_blocked([1, 2, 3, 4], [1, 1], 2)
        self.assertEqual(again.coefficients, good.coefficients)

    def test_inputs_are_not_mutated(self):
        a = [1, -2, 3, 0, 5] * 80   # 400 项
        b = [2, 0, -1] * 100        # 300 项
        a_snapshot, b_snapshot = list(a), list(b)
        convolve_blocked(a, b, 64)
        self.assertEqual((a, b), (a_snapshot, b_snapshot))
        with self.assertRaises(ExactConvolutionError):
            convolve_blocked(a, b + [10 ** 6 + 1], 64)
        self.assertEqual((a, b), (a_snapshot, b_snapshot))

    def test_generators_accepted_long(self):
        # 生成器在入口被物化为列表后切块，结果一致。
        result = convolve_blocked((i % 3 - 1 for i in range(500)),
                                  (1 if j % 2 else -1 for j in range(333)), 50)
        a = [i % 3 - 1 for i in range(500)]
        b = [1 if j % 2 else -1 for j in range(333)]
        self.assertEqual(result.coefficients, naive_convolve(a, b))


if __name__ == "__main__":
    unittest.main()
