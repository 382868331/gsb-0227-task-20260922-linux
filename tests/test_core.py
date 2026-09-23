"""NTT / CRT 原语与随机正确性（对照独立朴素参考）。"""

import random
import unittest

from exact_conv.ntt import ntt, convolution_mod
from exact_conv.crt import reconstruct
from exact_conv.ntt import MODULI

from tests.reference import naive_convolve


class NttTests(unittest.TestCase):
    def test_forward_inverse_roundtrip(self):
        # 补充性质：NTT 正变换->逆变换恢复原序列（仅作补充验证，期望值用
        # 普通整数构造，不依赖被测代码）。
        rng = random.Random(20260923)
        for mod in MODULI:
            data = [rng.randrange(mod) for _ in range(16)]
            back = ntt(ntt(data, 16, mod), 16, mod, inverse=True)
            self.assertEqual(back[:16], data)

    def test_convolution_mod_matches_definition(self):
        # 每个单模卷积都应与“逐系数取模的朴素卷积”一致。
        rng = random.Random(4242)
        a = [rng.randrange(-5000, 5000) for _ in range(13)]
        b = [rng.randrange(-5000, 5000) for _ in range(19)]
        expect = [0] * (13 + 19 - 1)
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                expect[i + j] = (expect[i + j] + x * y) % 998244353
        got = convolution_mod(a, b, 64, 998244353)
        self.assertEqual(got, expect)

    def test_convolution_mod_with_stride_not_power_of_two_input(self):
        # 长度 5+5-1=9 -> 16 点，验证非 2 幂输入长度下也正确。
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        expect = [x % 469762049 for x in naive_convolve(a, b)]
        got = convolution_mod(a, b, 16, 469762049)
        self.assertEqual(got, expect)


class CrtTests(unittest.TestCase):
    def test_reconstruct_direct_check(self):
        # 直接构造已知值验证 Garner 折叠：正负、0、边界附近。
        moduli = (998244353, 1004535809)
        for value in [0, 1, -1, 123456789, -123456789,
                      (998244353 * 1004535809) // 2 - 3,
                      -((998244353 * 1004535809) // 2 - 3)]:
            residues = [value % p for p in moduli]
            self.assertEqual(reconstruct(residues, moduli), value)

    def test_reconstruct_single_modulus(self):
        p = 998244353
        self.assertEqual(reconstruct([0], (p,)), 0)
        self.assertEqual(reconstruct([7], (p,)), 7)
        # 单模对称区间 (-p/2, p/2]：端点两侧各取一点。
        v = p // 2
        self.assertEqual(reconstruct([v % p], (p,)), v)
        v = -(p // 2 - 5)
        self.assertEqual(reconstruct([v % p], (p,)), v)

    def test_reconstruct_three_moduli_random(self):
        rng = random.Random(99)
        moduli = tuple(MODULI)
        big_product = 998244353 * 1004535809 * 469762049
        for _ in range(50):
            value = rng.randrange(-big_product // 2 + 1, big_product // 2)
            residues = [value % p for p in moduli]
            self.assertEqual(reconstruct(residues, moduli), value)

    def test_reject_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            reconstruct([1], (7, 11))
        with self.assertRaises(ValueError):
            reconstruct([], ())


if __name__ == "__main__":
    unittest.main()
