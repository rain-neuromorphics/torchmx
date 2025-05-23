# Adopted from: https://github.com/pytorch/ao/blob/v0.6.1/torchao/prototype/mx_formats/constants.py

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, repr=False)
class DType:
    name: str
    # The maximum value representable by the dtype
    max: float
    # The maximum power of 2 that can be represented by the dtype (larget binade)
    max_pow2: int
    # Exponent bias
    exponent_bias: int
    # Exponent bits
    exponent_bits: int
    # Mantissa bits
    mantissa_bits: int
    # A flag indicating whether the dtype supports NaN
    has_nan: bool
    # A flag indicating whether the dtype supports infinity
    has_inf: bool
    # PyTorch dtype if present: torch.float8_e4m3fn,, etc.
    torch_dtype: Optional[torch.dtype] = None

    def __repr__(self):
        return self.name


# Supported element dtypes
float8_e4m3 = DType(
    name="float8_e4m3",
    max=448.0,
    max_pow2=8,
    exponent_bias=7,
    exponent_bits=4,
    mantissa_bits=3,
    torch_dtype=torch.float8_e4m3fn,
    has_nan=True,
    has_inf=False,
)

float6_e3m2 = DType(
    name="float6_e3m2",
    max=28.0,
    max_pow2=4,
    exponent_bias=3,
    exponent_bits=3,
    mantissa_bits=2,
    torch_dtype=None,
    has_nan=False,
    has_inf=False,
)

float6_e2m3 = DType(
    name="float6_e2m3",
    max=7.5,
    max_pow2=2,
    exponent_bias=1,
    exponent_bits=2,
    mantissa_bits=3,
    torch_dtype=None,
    has_nan=False,
    has_inf=False,
)

float4_e2m1 = DType(
    name="float4_e2m1",
    max=6.0,
    max_pow2=2,
    exponent_bias=1,
    exponent_bits=2,
    mantissa_bits=1,
    torch_dtype=None,
    has_nan=False,
    has_inf=False,
)

int8 = DType(
    name="int8",
    max=127.0,
    max_pow2=6,
    exponent_bias=0,
    exponent_bits=0,
    mantissa_bits=7,
    torch_dtype=torch.int8,
    has_nan=False,
    has_inf=False,
)

float64 = DType(
    name="float64",
    max=torch.finfo(torch.float64).max,
    max_pow2=1023,
    exponent_bias=1023,
    exponent_bits=11,
    mantissa_bits=52,
    torch_dtype=torch.float64,
    has_nan=True,
    has_inf=True,
)

float32 = DType(
    name="float32",
    max=torch.finfo(torch.float32).max,
    max_pow2=127,
    exponent_bias=127,
    exponent_bits=8,
    mantissa_bits=23,
    torch_dtype=torch.float32,
    has_nan=True,
    has_inf=True,
)

bfloat16 = DType(
    name="bfloat16",
    max=torch.finfo(torch.bfloat16).max,
    max_pow2=127,
    exponent_bias=127,
    exponent_bits=8,
    mantissa_bits=7,
    torch_dtype=torch.bfloat16,
    has_nan=True,
    has_inf=True,
)


float22_e8m13 = DType(
    name="float22_e8m13",
    max=2**127 * (2 - 2**-13),
    max_pow2=127,
    exponent_bias=127,
    exponent_bits=8,
    mantissa_bits=13,
    has_nan=True,
    has_inf=True,
)

# Supported element dtypes
SUPPORTED_ELEM_DTYPES = (
    float8_e4m3,
    float6_e3m2,
    float6_e2m3,
    float4_e2m1,
    int8,
)

# Supported element dtypes
SUPPORTED_FP_ELEM_DTYPES = (
    float8_e4m3,
    float6_e3m2,
    float6_e2m3,
    float4_e2m1,
)


# Mapping from string to DType.
STR_TO_SUPPORTED_ELEM_DTYPE = {d.name: d for d in SUPPORTED_ELEM_DTYPES}

"""
Exponent E8M0 encoding details (OCP spec section 5.4.1):
  * bias: 127
  * supported exponent range: -127 to 127
  * infinities: N/A
  * NaN: 11111111
  * Zeros: N/A
"""

e8m0 = DType(
    name="e8m0",
    max=2**127,
    max_pow2=127,
    exponent_bias=127,
    exponent_bits=8,
    mantissa_bits=0,
    has_nan=True,
    has_inf=False,
)

E8M0_EXPONENT_NAN_VAL = 255
