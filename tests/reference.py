"""Independent naive integer convolution reference.

This module deliberately imports nothing from :mod:`exact_ntt`: expected
values in the test suite must come from a separate, obviously-correct
formulation

    C[k] = sum_{i+j=k} A[i] * B[j]

with schoolbook O(n*m) arithmetic on Python's arbitrary-precision ints.
Round-trip checks (transform then inverse) are used only as supplements,
never as the oracle.
"""

from __future__ import annotations


def naive_convolution(a: list[int], b: list[int]) -> list[int]:
    """Exact schoolbook convolution of two non-empty integer sequences."""
    if not a or not b:
        raise ValueError("naive reference requires non-empty operands")
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x == 0:
            continue
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out
