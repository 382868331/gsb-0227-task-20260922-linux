"""公开错误类型层次。

所有由本库主动抛出的错误都继承 :class:`ExactConvolutionError`
(它同时是 ``ValueError``)，调用方既可以按具体类型处理，也可以统一捕获。
"""


class ExactConvolutionError(ValueError):
    """本库所有错误的基类。"""


class InvalidPolynomialError(ExactConvolutionError):
    """输入不是合法的整数系数序列（结构/类型错误、空序列等）。"""


class InvalidCoefficientError(ExactConvolutionError):
    """某个系数不是可接受的精确整数（bool / float / NaN / Infinity / 非数值）。"""


class LengthOutOfBoundError(ExactConvolutionError):
    """多项式长度为空或超过接口上限 ``MAX_POLY_LENGTH = 256``。"""


class CoefficientBoundError(ExactConvolutionError):
    """某个系数绝对值超过接口上限 ``MAX_COEFF_ABS = 10**6``。

    本库只做*可证精确*的卷积：超界意味着结果系数可能落在模数乘积无法
    唯一区分的范围之外，因此直接拒绝，而不是静默取模冒充精确结果。
    """


class UnprovableBoundError(ExactConvolutionError):
    """固定模数集合的乘积不足以对该系数界保证 CRT 唯一恢复。

    在当前接口上限内（最多 256 项、系数绝对值 <= 10**6）该错误不会发生，
    它属于防御性错误：明确区分“证明无法成立”与其他运行期失败。
    """


class InvalidBlockSizeError(ExactConvolutionError):
    """固定块长入口的 ``block_size`` 非法。

    合法值为 ``1 .. MAX_POLY_LENGTH`` 的精确 ``int``（拒绝 bool、float、
    0、负数与超过单块上限的值——超过上限会使块本身无法被旧精确卷积接受）。
    """
