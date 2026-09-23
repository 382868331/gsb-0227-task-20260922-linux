"""Iterative Number Theoretic Transform over a fixed NTT prime.

The butterfly is written by hand (Cooley-Tukey, decimation-in-time with an
explicit bit-reversal permutation); no third-party FFT/NTT implementation is
used.  The inverse transform divides by the transform length (the modular
inverse of ``N``), as required.
"""

from __future__ import annotations

from .params import MAX_TRANSFORM_LENGTH, PRIMITIVE_ROOT


def _check_transform_length(n: int, p: int) -> None:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"NTT length must be a positive power of two, got {n}")
    limit = MAX_TRANSFORM_LENGTH[p]
    if n > limit:
        raise ValueError(
            f"NTT length {n} does not divide p-1 for p={p} (limit 2**{limit.bit_length()-1})"
        )


def ntt(a: list[int], p: int, invert: bool = False) -> list[int]:
    """In-place iterative NTT of length a power of two modulo ``p``.

    Returns the same list (permuted in place) for convenience.  When
    ``invert`` is true the inverse primitive root is used and every output
    is multiplied by ``N^{-1} mod p`` (the "inverse length" step).
    """
    n = len(a)
    _check_transform_length(n, p)

    # ---- bit-reversal permutation ------------------------------------
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # ---- butterfly stages ---------------------------------------------
    length = 2
    while length <= n:
        # Primitive length-th root of unity (inverse root on the way back).
        wlen = pow(PRIMITIVE_ROOT, (p - 1) // length, p)
        if invert:
            wlen = pow(wlen, p - 2, p)
        half = length >> 1
        for start in range(0, n, length):
            w = 1
            end = start + half
            for i in range(start, end):
                u = a[i]
                v = a[i + half] * w % p
                a[i] = (u + v) % p
                a[i + half] = (u - v) % p
                w = w * wlen % p
        length <<= 1

    if invert:
        inv_n = pow(n, p - 2, p)
        for i in range(n):
            a[i] = a[i] * inv_n % p
    return a


def convolution_mod(a: list[int], b: list[int], size: int, p: int) -> list[int]:
    """Cyclic-free linear convolution of ``a`` and ``b`` modulo ``p``.

    Both sequences are zero-padded to ``size`` (a power of two at least
    ``len(a) + len(b) - 1``), transformed, multiplied pointwise and
    transformed back.  Negative inputs are reduced canonically to
    ``0 .. p-1`` *here*; callers working the exact path must only do so
    after the magnitude bound has been proved sufficient.
    """
    fa = [0] * size
    fb = [0] * size
    for i, x in enumerate(a):
        fa[i] = x % p
    for i, x in enumerate(b):
        fb[i] = x % p

    ntt(fa, p)
    ntt(fb, p)
    for i in range(size):
        fa[i] = fa[i] * fb[i] % p
    ntt(fa, p, invert=True)
    return fa


def next_power_of_two(x: int) -> int:
    """Smallest power of two ``>= x`` (x >= 1)."""
    return 1 << (x - 1).bit_length()
