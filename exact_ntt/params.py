"""Fixed public parameters for the exact multi-modulus NTT convolution.

Contract (see README):
    * integer coefficients satisfy |x| <= MAX_COEFF_ABS
    * each operand has length 1 <= len <= MAX_LENGTH
    * three fixed NTT-friendly primes, all with primitive root 3
"""

from __future__ import annotations

#: Maximum absolute value of any input coefficient.
MAX_COEFF_ABS: int = 10**6

#: Maximum number of coefficients in one operand.
MAX_LENGTH: int = 256

#: Common primitive root for every prime below.
PRIMITIVE_ROOT: int = 3

#: NTT primes, in the fixed order in which the modulus selector adds them.
#:
#:   998244353 = 119 * 2**23 + 1   (max NTT length 2**23)
#:  1004535809 = 479 * 2**21 + 1   (max NTT length 2**21)
#:   469762049 =   7 * 2**26 + 1   (max NTT length 2**26)
NTT_PRIMES: tuple[int, ...] = (998244353, 1004535809, 469762049)

#: Largest power-of-two transform length dividing p - 1 for each prime.
MAX_TRANSFORM_LENGTH: dict[int, int] = {
    998244353: 1 << 23,
    1004535809: 1 << 21,
    469762049: 1 << 26,
}
