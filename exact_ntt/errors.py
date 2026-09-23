"""Error semantics for the exact_ntt package.

All failures are *explicit*: the implementation never reduces an
out-of-contract integer modulo an NTT prime and presents the residue as an
exact result.  Every error message identifies the offending operand and,
where applicable, its index.
"""

from __future__ import annotations


class ExactConvolutionError(ValueError):
    """Base class for all contract violations in this package."""


class CoefficientError(ExactConvolutionError):
    """A coefficient is not a finite integer or exceeds the magnitude bound."""


class LengthError_(ExactConvolutionError):
    """An operand is empty or longer than ``MAX_LENGTH``.

    A dedicated subclass (rather than the builtin :class:`ValueError`) keeps
    the error locatable to the operand length.
    """


class ExactRecoveryError(ExactConvolutionError):
    """The exact result is provably unreachable with the available primes.

    This means the proved coefficient bound ``B`` does not satisfy
    ``2 * B < M`` where ``M`` is the product of all three NTT primes.  No
    further modulus exists to add, so raising is mathematically mandatory:
    returning a CRT residue would silently alias an out-of-range integer.
    """
