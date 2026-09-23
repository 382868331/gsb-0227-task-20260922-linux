"""Exact integer polynomial convolution via multi-modulus NTT + CRT.

Public entry point: :func:`convolve_exact`.  The pipeline is

1. validate both operands (locatable errors, no silent truncation);
2. prove the coefficient bound ``B = min(n,m) * max|A| * max|B|``;
3. choose the shortest prefix of the three fixed primes whose product
   ``M`` satisfies ``M > 2 * B``;
4. run a hand-written iterative NTT under each chosen prime;
5. CRT-reconstruct every coefficient into the symmetric integer range.

All operations are pure and stateless: an invalid second operand raises
before any NTT work and no partial user-visible state is ever produced.
"""

from __future__ import annotations

from dataclasses import dataclass

from .crt import reconstruct, select_moduli
from .errors import ExactRecoveryError
from .ntt import convolution_mod, next_power_of_two
from .params import NTT_PRIMES
from .validation import coefficient_bound, validate_operand

__all__ = ["ConvolutionResult", "convolve_exact", "integer_convolution_info"]


@dataclass(frozen=True)
class ConvolutionResult:
    """Exact convolution output together with its proof metadata.

    Attributes:
        coefficients: the exact integer convolution, length
            ``len(a) + len(b) - 1``.
        bound: proved worst-case bound ``min(n,m)*max|a|*max|b|``.
        moduli: NTT primes actually used (shortest prefix with
            ``product > 2 * bound``).
        transform_length: power-of-two NTT length (zero-padded).
    """

    coefficients: list[int]
    bound: int
    moduli: tuple[int, ...]
    transform_length: int

    @property
    def modulus_product(self) -> int:
        """Product of the moduli used; strictly greater than ``2*bound``."""
        product = 1
        for p in self.moduli:
            product *= p
        return product


@dataclass(frozen=True)
class _Plan:
    bound: int
    moduli: tuple[int, ...]
    size: int
    out_len: int


def integer_convolution_info(length_a: int, length_b: int,
                             max_abs_a: int, max_abs_b: int) -> dict[str, object]:
    """Report which moduli *would* be used for given shapes/bounds.

    Pure planning helper (does no transform).  Inputs are checked with the
    same integer rules as the operands themselves.
    """
    for name, value in (("length_a", length_a), ("length_b", length_b),
                        ("max_abs_a", max_abs_a), ("max_abs_b", max_abs_b)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    for name, value in (("length_a", length_a), ("length_b", length_b)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    for name, value in (("max_abs_a", max_abs_a), ("max_abs_b", max_abs_b)):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    bound = min(length_a, length_b) * max_abs_a * max_abs_b
    moduli = select_moduli(bound)
    product = 1
    for p in moduli:
        product *= p
    out_len = length_a + length_b - 1
    return {
        "bound": bound,
        "moduli": moduli,
        "modulus_count": len(moduli),
        "modulus_product": product,
        "twice_bound": 2 * bound,
        "transform_length": next_power_of_two(out_len),
        "output_length": out_len,
    }


def _make_plan(a: list[int], b: list[int]) -> _Plan:
    bound = coefficient_bound(a, b)
    moduli = select_moduli(bound)  # raises ExactRecoveryError if provably impossible
    out_len = len(a) + len(b) - 1
    size = next_power_of_two(out_len)
    return _Plan(bound=bound, moduli=moduli, size=size, out_len=out_len)


def convolve_exact(a: object, b: object) -> ConvolutionResult:
    """Exact integer polynomial multiplication.

    Args:
        a: sequence of ``int`` coefficients, 1..256 elements,
            each ``|a_i| <= 10**6``.
        b: same contract.

    Returns:
        :class:`ConvolutionResult` with the exact convolution and proof
        metadata.

    Raises:
        CoefficientError: non-integer / bool / NaN-or-infinity float /
            coefficient outside the bound.
        LengthError_: empty operand, wrong type, or length > 256.
        ExactRecoveryError: all three primes together cannot separate the
            symmetric range (cannot occur for in-contract inputs, but kept
            as an explicit guard instead of silent aliasing).
    """
    a_ints = validate_operand(a, "a")
    b_ints = validate_operand(b, "b")
    plan = _make_plan(a_ints, b_ints)

    # One residue vector per chosen prime; each NTT call works on its own
    # freshly padded lists, so a failure mid-way leaves nothing half-written.
    residue_columns: list[list[int]] = [
        convolution_mod(a_ints, b_ints, plan.size, p)[: plan.out_len]
        for p in plan.moduli
    ]

    coefficients: list[int] = []
    for k in range(plan.out_len):
        column = [rv[k] for rv in residue_columns]
        coefficients.append(reconstruct(column, plan.moduli, plan.bound))

    return ConvolutionResult(
        coefficients=coefficients,
        bound=plan.bound,
        moduli=plan.moduli,
        transform_length=plan.size,
    )
