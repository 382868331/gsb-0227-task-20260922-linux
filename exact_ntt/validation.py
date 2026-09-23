"""Input validation with locatable, non-silent errors.

Rules (fixed public contract):

* Coefficients must be genuine integers.  ``bool`` is rejected even though
  ``isinstance(True, int)`` is true in Python, and ``float`` / other
  ``numbers.Real`` instances are rejected rather than truncated.  NaN and
  infinity are therefore impossible to smuggle in, but the checks report
  them with an explicit message if a floating value is ever passed.
* Sequences must be non-empty lists/tuples of length at most
  ``MAX_LENGTH``.  Bare iterators/generators are rejected: a length-bound
  contract cannot be checked on a one-shot stream without consuming it.
* Every coefficient must satisfy ``|a_i| <= MAX_COEFF_ABS``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .errors import CoefficientError, LengthError_
from .params import MAX_COEFF_ABS, MAX_LENGTH


def _check_coefficient(value: Any, operand: str, index: int) -> int:
    """Validate one coefficient; return it as a plain ``int``."""
    where = f"{operand}[{index}]"
    if isinstance(value, bool):
        raise CoefficientError(
            f"{where}: bool is not accepted as an integer coefficient (got {value!r})"
        )
    if isinstance(value, int):
        ivalue = value
    elif isinstance(value, float):
        # Explicit, locatable handling instead of int() truncation.
        if math.isnan(value):
            raise CoefficientError(f"{where}: NaN is not a valid coefficient")
        if math.isinf(value):
            raise CoefficientError(
                f"{where}: infinity is not a valid coefficient (got {value!r})"
            )
        raise CoefficientError(
            f"{where}: coefficient must be int, got float ({value!r}); "
            "floating-point values are never truncated"
        )
    else:
        raise CoefficientError(
            f"{where}: coefficient must be int, got {type(value).__name__} ({value!r})"
        )
    if abs(ivalue) > MAX_COEFF_ABS:
        raise CoefficientError(
            f"{where}: coefficient {ivalue} exceeds the proved bound "
            f"|x| <= {MAX_COEFF_ABS}"
        )
    return ivalue


def validate_operand(operand: Any, name: str) -> list[int]:
    """Validate one input sequence and return a defensive copy of ints.

    The copy guarantees that later mutation of the caller's list cannot
    affect an in-flight convolution (operations are stateless overall).
    """
    if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes)):
        raise LengthError_(
            f"{name}: expected a list/tuple of integers, "
            f"got {type(operand).__name__}"
        )
    n = len(operand)
    if n == 0:
        raise LengthError_(f"{name}: operand must be non-empty")
    if n > MAX_LENGTH:
        raise LengthError_(
            f"{name}: length {n} exceeds the interface bound MAX_LENGTH={MAX_LENGTH}"
        )
    return [_check_coefficient(v, name, i) for i, v in enumerate(operand)]


def coefficient_bound(a: Sequence[int], b: Sequence[int]) -> int:
    r"""Proved worst-case absolute bound on any output coefficient.

    For ``C_k = sum_i A_i B_{k-i}`` every coefficient is a sum of at most
    ``min(len(a), len(b))`` products with ``|A_i B_j| <= Amax * Bmax``, so

        .. math:: |C_k| <= B := min(n, m) * Amax * Bmax.

    The bound is a *proof* bound (not a measured one): the CRT only needs to
    cover every value the convolution could take under the input contract.
    """
    amax = max((abs(x) for x in a), default=0)
    bmax = max((abs(x) for x in b), default=0)
    return min(len(a), len(b)) * amax * bmax
