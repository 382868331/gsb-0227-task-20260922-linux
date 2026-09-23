"""exact_ntt — exact integer convolution by multi-modulus NTT and CRT.

Public API:

* :func:`convolve_exact` — exact integer polynomial multiplication;
* :class:`ConvolutionResult` — output plus proof metadata;
* :func:`integer_convolution_info` — modulus-selection planner;
* error types :class:`ExactConvolutionError`, :class:`CoefficientError`,
  :class:`LengthError_`, :class:`ExactRecoveryError`;
* constants :data:`NTT_PRIMES`, :data:`PRIMITIVE_ROOT`,
  :data:`MAX_COEFF_ABS`, :data:`MAX_LENGTH`.
"""

from __future__ import annotations

from .convolution import ConvolutionResult, convolve_exact, integer_convolution_info
from .errors import (
    CoefficientError,
    ExactConvolutionError,
    ExactRecoveryError,
    LengthError_,
)
from .params import (
    MAX_COEFF_ABS,
    MAX_LENGTH,
    NTT_PRIMES,
    PRIMITIVE_ROOT,
)

__all__ = [
    "ConvolutionResult",
    "convolve_exact",
    "integer_convolution_info",
    "ExactConvolutionError",
    "CoefficientError",
    "LengthError_",
    "ExactRecoveryError",
    "MAX_COEFF_ABS",
    "MAX_LENGTH",
    "NTT_PRIMES",
    "PRIMITIVE_ROOT",
]
