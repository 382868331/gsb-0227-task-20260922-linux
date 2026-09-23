"""Fixed-input offline demonstration.

Run with ``python demo.py``.  Prints, for deterministic hard-coded inputs:

1. a small exact convolution using a single modulus;
2. a large-magnitude convolution that switches to two moduli;
3. a sparse convolution with long zero tails (still one modulus);
4. a boundary case that is genuinely rejected with a locatable error.

Nothing is pre-recorded: every printed result is computed by the NTT/CRT
core and independently cross-checked with schoolbook arithmetic embedded in
this file.
"""

from __future__ import annotations

import time

from exact_ntt import (
    MAX_COEFF_ABS,
    CoefficientError,
    convolve_exact,
    integer_convolution_info,
)


def schoolbook(a, b):
    """Local independent check used only to annotate the demo output."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def show_case(title, a, b):
    print("=" * 72)
    print(title)
    print(f"  len(a)={len(a)}, len(b)={len(b)}")
    print(f"  max|a|={max(abs(x) for x in a)}, max|b|={max(abs(x) for x in b)}")

    info = integer_convolution_info(
        len(a), len(b), max(abs(x) for x in a), max(abs(x) for x in b)
    )
    print(f"  proved bound B = min(n,m)*max|a|*max|b| = {info['bound']}")
    print(f"  need modulus product > 2B = {info['twice_bound']}")
    print(f"  selected {info['modulus_count']} modulus(es): {info['moduli']}")
    print(f"  modulus product = {info['modulus_product']}")
    print(f"  NTT length (zero-padded power of two) = {info['transform_length']}")

    result = convolve_exact(a, b)
    assert result.moduli == info["moduli"]
    assert result.modulus_product > 2 * result.bound

    expected = schoolbook(a, b)
    assert result.coefficients == expected, "demo cross-check failed"
    print(f"  exact convolution ({len(result.coefficients)} coefficients):")
    print(f"    {result.coefficients}")
    print("  cross-checked against independent schoolbook arithmetic: OK")


def main() -> None:
    start = time.perf_counter()

    # Case 1: small magnitudes -> bound 54, one modulus suffices.
    show_case(
        "CASE 1 (single modulus): (1 + 2x - 3x^2 + x^3) * (4 - 5x + 6x^2)",
        [1, 2, -3, 1],
        [4, -5, 6],
    )

    # Case 2: one product of maximal coefficients already reaches 10^12,
    # far beyond P1/2 ~ 4.99e8, so two moduli are selected despite the
    # short operands.
    show_case(
        "CASE 2 (multi-modulus switch): tiny operands, |coefficients|=10^6",
        [MAX_COEFF_ABS, -MAX_COEFF_ABS, MAX_COEFF_ABS],
        [MAX_COEFF_ABS, MAX_COEFF_ABS],
    )

    # Case 3: long zero tails on both sides; proved bound is small because
    # the actual magnitudes are small, so one modulus is still enough.
    show_case(
        "CASE 3 (long zero tails): sparse 17x13 operands",
        [7, -3, 0, 11] + [0] * 13,
        [2, 0, -9, 0, 5] + [0] * 8,
    )

    # Case 4: an actual rejection boundary.  One coefficient exceeds the
    # proved interface bound by exactly 1; the call must raise rather than
    # reduce it modulo an NTT prime and masquerade as exact.
    print("=" * 72)
    print("CASE 4 (rejected boundary): coefficient 10^6 + 1 is out of contract")
    bad_a = [MAX_COEFF_ABS + 1, 2, -3]
    good_b = [7, -1]
    try:
        convolve_exact(bad_a, good_b)
    except CoefficientError as exc:
        print(f"  raised {type(exc).__name__}: {exc}")
        print("  -> refused: out-of-contract inputs are never reduced modulo a prime")
    else:  # pragma: no cover - would indicate a broken core
        raise AssertionError("expected CoefficientError was not raised")

    elapsed = time.perf_counter() - start
    print("=" * 72)
    print(f"demo finished in {elapsed:.3f}s (standard library only, offline)")


if __name__ == "__main__":
    main()
