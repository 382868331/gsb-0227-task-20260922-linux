"""Chinese Remainder Theorem reconstruction into symmetric integers.

Residues are combined incrementally with Garner's formula.  If ``t`` is the
canonical reconstruction modulo a running product ``M`` and a new residue
``r`` modulo an odd prime ``p`` is added, then

    t' = t + M * ((r - t) * M^{-1}) mod p,   M' = M * p

is the unique value in ``[0, M')`` matching every residue.  At the end the
canonical representative is mapped to the symmetric interval
``[-(M-1)/2, (M-1)/2]``; exact recovery of a coefficient is guaranteed
whenever ``|c| <= B`` with ``M > 2 * B``.
"""

from __future__ import annotations

from .errors import ExactRecoveryError
from .params import NTT_PRIMES


def select_moduli(bound: int) -> tuple[int, ...]:
    """Return the shortest prefix of the fixed primes with product > 2*bound.

    The prefix order is fixed (:data:`params.NTT_PRIMES`); adding primes one
    at a time is what makes single-/multi-modulus switching depend only on
    the proved coefficient bound.  At least one prime is always used, even
    for ``bound == 0``, so the transform/CRT path stays uniform.

    Raises :class:`ExactRecoveryError` when even the product of all three
    primes is insufficient -- a *proved impossibility* for this contract,
    never a budget/time-out condition.
    """
    if bound < 0:
        raise ValueError(f"bound must be non-negative, got {bound}")
    chosen: list[int] = []
    product = 1
    for p in NTT_PRIMES:
        chosen.append(p)
        product *= p
        if product > 2 * bound:
            return tuple(chosen)
    raise ExactRecoveryError(
        f"no exact reconstruction possible: proved bound |c| <= {bound} "
        f"requires modulus product > {2 * bound}, but the product of all "
        f"{len(NTT_PRIMES)} fixed primes is only {product}"
    )


def garner_extend(t: int, modulus: int, residue: int, p: int) -> int:
    """Extend a canonical CRT representative ``t`` (mod ``modulus``) by one prime."""
    delta = ((residue - t) % p) * pow(modulus, -1, p) % p
    return t + modulus * delta


def reconstruct(residues: list[int], primes: tuple[int, ...], bound: int) -> int:
    """Reconstruct one exact integer from its residues, with a range proof.

    ``residues[i]`` is the value modulo ``primes[i]``.  Raises
    :class:`ExactRecoveryError` if the combined modulus does not cover
    ``2 * bound + 1`` distinct symmetric values.
    """
    t = residues[0] % primes[0]
    modulus = primes[0]
    for r, p in zip(residues[1:], primes[1:]):
        t = garner_extend(t, modulus, r % p, p)
        modulus *= p
    if modulus <= 2 * bound:
        # Defensive: select_moduli() normally prevents reaching this state.
        raise ExactRecoveryError(
            f"combined modulus {modulus} does not exceed twice the proved "
            f"bound {2 * bound}"
        )
    if t > modulus // 2:  # modulus is odd: symmetric range is [-M//2, M//2]
        t -= modulus
    if abs(t) > bound:
        # Can only happen if an input slipped past validation; reporting it
        # is still better than returning a wrong "exact" integer.
        raise ExactRecoveryError(
            f"reconstructed coefficient {t} lies outside the proved bound "
            f"|c| <= {bound}"
        )
    return t
