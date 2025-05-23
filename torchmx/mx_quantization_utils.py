from typing import Optional

import torch
from torchao.prototype.mx_formats.custom_cast import (
    f32_to_f4_unpacked,
    f32_to_f6_e2m3_unpacked,
    f32_to_f6_e3m2_unpacked,
)

from . import dtypes
from .utils import pack_uint4, unpack_uint4

# TODO: add unit test for the file


def unpack_bfloat16(
    x: torch.Tensor, dtype: torch.dtype = torch.uint8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract the sign, exponent, and mantissa from a bfloat16 tensor

    Args:
    x: torch.Tensor, the input bfloat16 tensor
    dtype: torch.dtype, the dtype to cast the output tensors to. Default is torch.uint8

    Returns:
        sign: torch.Tensor, the sign of the tensor in uint8
        exponent: torch.Tensor, the exponent of the tensor in uint8
        mantissa: torch.Tensor, the mantissa of the tensor in uint8
    """

    assert (
        x.dtype == torch.bfloat16
    ), "x nust be of dtype torch.bfloat16 but got {x.dtype}"
    # Convert the bfloat16 tensor to a 16-bit integer tensor
    int_tensor = x.view(torch.int16)
    # Get the sign of the tensor
    sign = ((1 - int_tensor.sign()) // 2).to(dtype)

    # Extract the exponent (8 bits)
    exponent = (int_tensor >> dtypes.bfloat16.mantissa_bits & 0xFF).to(
        dtype
    )  # 8 bits, stored in uint8

    # Extract the mantissa (7 bits)
    mantissa = (int_tensor & 0x7F).to(dtype)  # 7 bits, stored in uint8

    return sign, exponent, mantissa


def unpack_fp32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpacks the given FP32 tensor to its components.

    Args:
        x (torch.Tensor): The packed FP32 tensor.

    Returns:
        sign (torch.Tensor): The sign bit tensor.
        exponent (torch.Tensor): The exponent tensor.
        mantissa (torch.Tensor): The mantissa tensor..
    """
    assert x.dtype == torch.float32
    x_int = x.view(torch.int32)
    x_sign = ((1 - x_int.sign()) // 2).to(torch.uint8)
    x_exp = (x_int >> 23 & 0xFF).to(torch.uint8)
    x_mantissa = (x_int & 0x7FFFFF).to(torch.int32)

    return x_sign, x_exp, x_mantissa


def unpack_fp64(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unpacks the given FP64 tensor to its components.

    Args:
        x (torch.Tensor): The packed FP64 tensor.

    Returns:
        sign (torch.Tensor): The sign bit tensor.
        exponent (torch.Tensor): The exponent tensor.
        mantissa (torch.Tensor): The mantissa tensor..
    """
    assert x.dtype == torch.float64
    x_int = x.view(torch.int64)
    x_sign = ((1 - x_int.sign()) // 2).to(torch.uint8)
    x_exp = (x_int >> 52 & 0x7FF).to(torch.int16)
    x_mantissa = (x_int & 0xFFFFFFFFFFFFF).to(torch.int64)

    return x_sign, x_exp, x_mantissa


def dequantize_to_dtype(
    data_lp: torch.Tensor,
    elem_dtype: dtypes.DType,
    target_dtype: torch.dtype,
    packing_dim: int = -1,
    is_packed_fp4: bool = True,
) -> torch.Tensor:
    """
    Dequantizes a elem_dtype packed as unit8 to the target_dtype by using an intermediate bfloa16 representation.
    Args:
        data_lp (torch.Tensor): The input tensor in low-precision format (must be of dtype torch.uint8).
        elem_dtype (dtypes.DType): The input element data type
        target_dtype (torch.dtype): The target data type to which the tensor will be dequantized.
        is_packed_fp4 (bool, optional): A flag indicating whether the input tensor is packed in FP4
        format. Defaults to True.
        packing_dim (int): The dimension along which the uint4 data is packed, default is -1.

    Returns:
        torch.Tensor: The dequantized tensor in the specified target data type.
    Raises:
        AssertionError: If the element data type is not supported or if the input tensor is not of dtype torch.uint8.
    """
    assert (
        elem_dtype in dtypes.SUPPORTED_FP_ELEM_DTYPES
    ), f"Unsupported dtype {elem_dtype}. Supported: {dtypes.SUPPORTED_FP_ELEM_DTYPES}"
    assert data_lp.dtype == torch.uint8, "Input tensor must be of dtype torch.uint8"
    if elem_dtype == dtypes.float8_e4m3:
        return data_lp.view(torch.float8_e4m3fn).to(target_dtype)

    if is_packed_fp4 and elem_dtype == dtypes.float4_e2m1:
        data_lp = unpack_uint4(data_lp, packing_dim)
    # Extract the exponent
    x_exp_unbiased = data_lp >> elem_dtype.mantissa_bits & (
        n_ones(elem_dtype.exponent_bits)
    )
    # Extract the mantissa
    x_mantissa = data_lp & n_ones(elem_dtype.mantissa_bits)
    # Extract the sign
    x_sign = data_lp >> (elem_dtype.mantissa_bits + elem_dtype.exponent_bits)

    # Convert the sign bit to {-1,1}
    x_sign_bfloat16 = 1 - 2 * x_sign.bfloat16()
    # Construct a bfloat16 tesnor as it can represent all FP8/6/4 formats
    mantissa_bfloat16 = x_mantissa.bfloat16() / (2**elem_dtype.mantissa_bits)
    mantissa_bfloat16 = torch.where(
        x_exp_unbiased == 0, mantissa_bfloat16, 1 + mantissa_bfloat16
    )
    exponent_int8 = (
        torch.where(x_exp_unbiased == 0, 1, x_exp_unbiased).to(torch.int8)
        - elem_dtype.exponent_bias
    )
    y = x_sign_bfloat16 * torch.pow(2.0, exponent_int8).bfloat16() * mantissa_bfloat16

    return y.to(target_dtype)


def round_to_even(
    mantissa: torch.Tensor, mantissa_shift: torch.Tensor | int
) -> torch.Tensor:
    """
    Round a mantissa to the nearest even value using a tensor of shift values.

    Args:
        mantissa (torch.Tensor) : A tensor containing the mantissa values to be rounded.
        mantissa_shift (torch.Tensor | int):   A tensor containing the shift values
            to be applied to each element of the `mantissa` tensor.
            The size of the `mantissa_shift` tensor should match the size of the `mantissa` tensor.
            Alternatively, a single integer value can be provided, in which case the same shift value
            will be applied to all elements of the `mantissa` tensor.

    Returns:
    torch.Tensor
        A tensor containing the mantissa values rounded to the nearest even value.
        The size of the output tensor will match the input `mantissa` tensor.

    Notes
    -----
    - The rounding follows the "round half to even" rule, where if the value
      to be discarded is exactly halfway between two integers, the result is rounded
      to the nearest even number.
    - This function supports element-wise operations, where the shifting is applied
      to each element of the mantissa according to the corresponding value in `mantissa_shift`.

    Examples
    --------
    >>> mantissa = torch.tensor([0b1010011, 0b1101101], dtype=torch.int32)
    >>> mantissa_shift = torch.tensor([2, 3], dtype=torch.int32)
    >>> round_to_even(mantissa, mantissa_shift)
    tensor([41, 27])
    """
    # Retain only the top mantissa bits by element-wise shifting
    reduced_mantissa = mantissa >> mantissa_shift  # Element-wise right shift

    # First, extract the remainder bits that were removed by the shift
    remainder = mantissa & (
        (1 << mantissa_shift) - 1
    )  # Element-wise mask for remainder

    # Extract the round bit, which is the most significant bit in the remainder
    round_bit = remainder >> (
        mantissa_shift - 1
    )  # Element-wise right shift to get the round bit

    # Rounding condition, round to even if:
    # 1. If the round bin is set (i.e., round_bit > 0) and,
    # 2. The current new_mantissa is odd  OR there are remaining non-zero bits in the remainder
    round_bit_check = round_bit > 0  # round bit is set
    oddity_check = reduced_mantissa % 2 == 1  # new_mantissa is odd
    remainder_check = (
        remainder & ((1 << (mantissa_shift - 1)) - 1)
    ).bool()  # there are remaining non-zero bits in the remainder

    # Create the rounding condition tensor-wise
    round_condition = torch.where(
        round_bit_check & (oddity_check | remainder_check),
        torch.ones_like(reduced_mantissa),  # Set to 1 where rounding is needed
        torch.zeros_like(reduced_mantissa),  # Set to 0 where rounding is not needed
    )

    # Add rounding to the mantissa where the condition is true
    rounded_mantissa = reduced_mantissa + round_condition

    return rounded_mantissa


def n_ones(n: int) -> int:
    """
    Returns a number with n ones in binary representation.
    for example: _n_ones(3) = 0b111 = 7
    """

    return (1 << n) - 1


def leading_one_position(mantissa: torch.Tensor, mantissa_size: int = 7):
    """
    Determine the position of the leading one bit in each element of the input tensor with LBS at
    position 0. If there is no 1 in the mantissa, the function returns -1.
    Args:
        mantissa (torch.Tensor): A tensor containing the mantissa values to be analyzed.
                                 Each element should be an integer.
    Returns:
        torch.Tensor: the position of the leading one bit in each element of the input tensor.

    """
    leading_one_positions = torch.full_like(mantissa, fill_value=-1)
    # Iterate over all bit positions (from 6 down to 0) to find the leading one
    for i in range(mantissa_size - 1, -1, -1):
        # Check if the current bit is set (i.e., mantissa & (1 << i) is non-zero)
        mask = (mantissa & (1 << i)) != 0

        # Set the position for those mantissas where the bit is set and the position
        # hasn't been set ye
        leading_one_positions = torch.where(
            mask & (leading_one_positions == -1), i, leading_one_positions
        )

    return leading_one_positions


def quantize_mx_with_e8m0_shared_exponent_hw_exact(
    data_hp: torch.Tensor,
    elem_dtype: dtypes.DType,
    shared_exponent: torch.Tensor,
    orig_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    A hardware-exact MX quantization function that handles the division and conversion
    to that target element data type explicitly.

    Args:
        data_hp (torch.Tensor): The high precision input tensor, (dtype=torch.bfloat16).
        elem_dtype (dtypes.DType): The target element data type for quantization.
        shared_exponent (torch.Tensor): The E8M0 scale shared exponent (dtype=torch.uint8).
        orig_shape (torch.Size): The original shape of the input tensor, used to reshape the output
        tensor. Optional, defaults to None.

    Returns:
        torch.Tensor (dtype=torch.uint8): The quantized tensor in the target lower precision format.

    Raises:
        AssertionError: If the provided elem_dtype is not supported.
    """
    assert data_hp.dtype == torch.bfloat16, "Only bfloat16 is supported"
    assert (
        shared_exponent.dtype == torch.uint8
    ), "`shared_exponent` must be of dtype torch.uint8"
    assert (
        elem_dtype in dtypes.SUPPORTED_FP_ELEM_DTYPES
    ), f"Unsupported dtype {elem_dtype}. Supported: {dtypes.SUPPORTED_FP_ELEM_DTYPES}"

    # Extract the sign, exponent, and mantissa from the high precision input tensor
    bf16_sign, bf16_exponent, bf16_mantissa = unpack_bfloat16(
        data_hp, dtype=torch.int16
    )
    # Make bf16_sign positive where the shared exponent is NaN
    bf16_sign = torch.where(
        shared_exponent == dtypes.E8M0_EXPONENT_NAN_VAL, 0, bf16_sign.to(torch.uint8)
    )

    zeros_mask = data_hp == 0
    # Step 1: Check if the input subnormal and normalize it
    subnormal_bf16_mask = (bf16_exponent == 0) & (~zeros_mask)
    # 1.1 Find the leading one in the mantissa
    leading_one = leading_one_position(bf16_mantissa)
    # Check leading ones is in the range [6, -1]
    assert (
        (leading_one <= 6) & (leading_one >= -1)
    ).all(), "Invalid leading one position"
    o_left_shifts = 7 - leading_one  # The of mantissa left shifts
    normalized_exponent = -(6 - leading_one)  # exponent correction factor

    # 1.2 Normalize the subnormal mantissa's
    normalized_mantissa = (bf16_mantissa << o_left_shifts) & 0x7F
    bf16_mantissa = torch.where(subnormal_bf16_mask, normalized_mantissa, bf16_mantissa)

    # 1.3 Correct the exponent
    bf16_exponent = torch.where(subnormal_bf16_mask, normalized_exponent, bf16_exponent)

    # Step 2: Calculate the new exponent
    new_exponent = (
        bf16_exponent.to(torch.int16) - shared_exponent + elem_dtype.exponent_bias
    )

    # Step 3: Mantissa rounding
    # The output mantissa is initialized with all zeros
    rounded_mantissa = torch.zeros_like(bf16_mantissa)

    # 3.1 Normal mantissa rounding with a constant shift, where new_exponent>0
    rounded_mantissa = torch.where(
        new_exponent > 0,
        round_to_even(
            bf16_mantissa, dtypes.bfloat16.mantissa_bits - elem_dtype.mantissa_bits
        ),
        rounded_mantissa,
    )

    # 3.2 Subnormal mantissa rounding, where  -el_data.mantissa_bits <=new_exponent <=0
    output_subnormal_mask = (
        (new_exponent <= 0) & (new_exponent >= -elem_dtype.mantissa_bits) & ~zeros_mask
    )

    # 3.2.1 Subnormalize the mantissa
    bf16_mantissa_3_msbs = bf16_mantissa >> 4  # the 3 most significant bits
    bf16_mantissa_4_lsbs = bf16_mantissa & 0xF  # the 4 least significant bits
    sticky_bit = (bf16_mantissa_4_lsbs != 0).to(torch.uint8)
    subnormalized_mantissa = 1 << 6 | bf16_mantissa_3_msbs << 3 | sticky_bit << 2

    mantissa_shift = (
        dtypes.bfloat16.mantissa_bits - elem_dtype.mantissa_bits - new_exponent
    )
    rounded_subnormalized_mantissa = round_to_even(
        subnormalized_mantissa, mantissa_shift
    )
    rounded_mantissa = torch.where(
        output_subnormal_mask, rounded_subnormalized_mantissa, rounded_mantissa
    )

    # 3.3 Check mantissa overflow
    mantissa_overflow_mask = rounded_mantissa > n_ones(elem_dtype.mantissa_bits)
    rounded_mantissa = torch.where(
        mantissa_overflow_mask, torch.zeros_like(rounded_mantissa), rounded_mantissa
    )
    new_exponent = torch.where(mantissa_overflow_mask, new_exponent + 1, new_exponent)

    # 3.4 Update subnormal mask
    output_subnormal_mask = (
        (new_exponent <= 0) & (new_exponent >= -elem_dtype.mantissa_bits) & ~zeros_mask
    )

    # Construct the empty output vector
    z = torch.empty_like(data_hp, dtype=torch.uint8)

    # Step 5: Underflow handling
    underflow_mask = (
        (new_exponent < -elem_dtype.mantissa_bits)
        | (shared_exponent.expand_as(new_exponent) == dtypes.E8M0_EXPONENT_NAN_VAL)
        | (data_hp == 0)
    )
    z = torch.where(underflow_mask, torch.zeros_like(z), z)

    # Step 4: Saturation handling
    saturation_mask = new_exponent > 2**elem_dtype.exponent_bits - 1
    max_normal_magnitude = n_ones(elem_dtype.mantissa_bits + elem_dtype.exponent_bits)
    if elem_dtype == dtypes.float8_e4m3:
        saturation_mask_extra = (new_exponent == 15) & (
            rounded_mantissa == 7
        )  # S_1111_111
        saturation_mask = saturation_mask | saturation_mask_extra
        max_normal_magnitude = 0b1111_110

    z = torch.where(saturation_mask, torch.full_like(z, max_normal_magnitude), z)

    # Step 5 construct subnormals
    z = torch.where(output_subnormal_mask, rounded_mantissa, z)

    # Step 6: Construct normal numbers
    normal_mask = ~(saturation_mask | underflow_mask | output_subnormal_mask)
    z = torch.where(
        normal_mask,
        new_exponent.clamp(1, 2**elem_dtype.exponent_bits - 1)
        << elem_dtype.mantissa_bits
        | rounded_mantissa,
        z,
    )

    # Step 7: Append the sign bt
    y = bf16_sign << (elem_dtype.mantissa_bits + elem_dtype.exponent_bits) | z.to(
        torch.uint8
    )

    # Reshape the output tensor to the original shape
    if orig_shape is not None:
        y = y.reshape(orig_shape)

    # Pack the 2 fp4 elements in one byte
    if elem_dtype == dtypes.float4_e2m1:
        y = pack_uint4(y)

    return y


def get_fp_scale(shared_exp_e8m0: torch.Tensor) -> torch.Tensor:
    """Takes the shared exponent of the MX scale, FP8(0-8-0), as a biased uint8 exponent

    Args:
        shared_exp_e8m0 (torch.Tensor): the shared exponent of the FP8(0-8-0) scale

    Returns:
        torch.Tensor: FP32 scale, 2**(shared_exponent - 127), with NaNs handling
    """
    # The scale is cast to FP32 to ensure that the division of the inputs by the MX scale is
    # done in FP32, to ensure no mantissa bits are lost from the BF16 input during the division.

    scale_fp = 2 ** (shared_exp_e8m0.float() - dtypes.e8m0.exponent_bias)
    scale_fp = torch.where(
        shared_exp_e8m0 == dtypes.E8M0_EXPONENT_NAN_VAL, float("nan"), scale_fp
    )

    return scale_fp


def quantize_mx_with_e8m0_shared_exponent_simulated(
    data_hp: torch.Tensor,
    elem_dtype: dtypes.DType,
    shared_exponent: torch.Tensor,
    orig_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    Simulated MX quantization function inspired by torchao. It accepts high precision input tensor
    (data_hp), the MX scale shared exponent (shared_exponent), and returns the quantized tensor
    in the elem_dtype. The steps are:
    1. normalize data_hp by performing a single-precision division with an MX scale in
           torch.float32
    2. quantize the normalized data_hp to the target elem_dtype by using the native torhcao function

    We call this implementation simulated because it is not an efficient hardware implementation

    Args:
        data_hp (torch.Tensor): The high precision input tensor, dtype is either torch.bfloat16 or torch.float
        elem_dtype (dtypes.DType): The target element data type for quantization.
        shared_exponent (torch.Tensor): The E8M0 scale shared exponent (dtype=torch.uint8).
        orig_shape (torch.Size): The original shape of the input tensor, used to reshape the output
        tensor. Optional, defaults to None.


    Returns:
        torch.Tensor (dtype=torch.uint8): The quantized tensor in the target lower precision format.

    Raises:
        AssertionError: If the provided elem_dtype is not supported.
    """
    # 1. convert the shared exponent to a FP32 scale
    scale_fp = get_fp_scale(shared_exponent)

    # 2. Divide by the MX-scale in high precision
    data_norm_hp = torch.clamp(
        data_hp / scale_fp, min=-1 * elem_dtype.max, max=elem_dtype.max
    )
    # flush to unsigned zero any block with NaN scale
    data_norm_hp = torch.where(data_norm_hp.isnan(), 0.0, data_norm_hp)

    if orig_shape is not None:
        data_norm_hp = data_norm_hp.reshape(orig_shape)

    # 3. cast to target dtype
    if elem_dtype == dtypes.float8_e4m3:
        # Casting back to torch.uint8 for consistency with the other float formats
        data_lp = data_norm_hp.to(torch.float8_e4m3fn).view(torch.uint8)
    elif elem_dtype == dtypes.float6_e2m3:
        data_lp = f32_to_f6_e2m3_unpacked(data_norm_hp)
    elif elem_dtype == dtypes.float6_e3m2:
        data_lp = f32_to_f6_e3m2_unpacked(data_norm_hp)
    elif elem_dtype == dtypes.float4_e2m1:
        data_lp = f32_to_f4_unpacked(data_norm_hp)
        data_lp = pack_uint4(data_lp)
    elif elem_dtype == dtypes.int8:
        # torch.round is needed because, tensor.to(torch.int8) doesn't do what we want
        # >>> a = torch.tensor(126.7)
        # >>> a.to(torch.int8)
        # tensor(126, dtype=torch.int8)
        # >>> torch.round(a).to(torch.int8)
        # tensor(127, dtype=torch.int8)
        data_lp = torch.round(data_norm_hp).to(elem_dtype.torch_dtype)
    else:
        raise AssertionError(f"unsupported, dtype: {elem_dtype}")
    return data_lp


def get_e8m0_shared_exponent(
    data_hp: torch.Tensor, elem_dtype: dtypes.DType
) -> torch.Tensor:
    """
    Computes the shared exponent for a given high-precision tensor.
    Args:
        data_hp (torch.Tensor): High-precision input tensor, with block size as the last dimension.
        dtype must be torch.bfloat16 or torch.float.
        elem_dtype (dtypes.DType):  target element dtype
    Returns:
        torch.Tensor: MX-scale exponent tensor as torch.uint8
    """
    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    assert (
        elem_dtype in dtypes.SUPPORTED_ELEM_DTYPES
    ), f"Unsupported dtype {elem_dtype}. Supported: {dtypes.SUPPORTED_ELEM_DTYPES}"

    # Extract the 8-bit exponent from the high-precision input tensor
    if data_hp.dtype == torch.bfloat16:
        # viewing a torch.bfloat16 as torch.int16 converts the FP value to a signed 16-bit integer:
        # (-1)**sign_bit * 2**(exponent - 127) * mantissa -> (-1)**sign_bit * ((exponent << 7) + mantissa)
        # For example, for x_fp = - 2**-13 * 1.125 -> x_int = - ((114 << 7) + 16) = - 14608
        data_int = data_hp.view(torch.int16)
        # Extract the exponent (8 bits)
        data_exponent = (data_int >> dtypes.bfloat16.mantissa_bits & 0xFF).to(
            torch.uint8
        )
    else:
        # viewing a torch.float32 as torch.int32 converts the FP value to a signed 32-bit integer:
        # (-1)**sign_bit * 2**(exponent - 127) * mantissa -> (-1)**sign_bit * ((exponent << 23) + mantissa)
        # For example, for x_fp = - 2**-13 * 1.125 -> x_int = - ((114 << 32) + 16) = - 957349888
        data_int = data_hp.view(torch.int32)
        # Extract the exponent (8 bits)
        data_exponent = (data_int >> dtypes.float32.mantissa_bits & 0xFF).to(
            torch.uint8
        )
    # get the largest exponent in the input tensor
    max_exponent = torch.amax(data_exponent, dim=-1)

    # calculate the biased shared exponent
    e8m0_max_biased_exponent = dtypes.e8m0.exponent_bias + dtypes.e8m0.max_pow2  # 254
    shared_exponent = max_exponent.to(torch.int16) - elem_dtype.max_pow2
    shared_exponent = torch.clamp(shared_exponent, 0, e8m0_max_biased_exponent).to(
        torch.uint8
    )

    # saturation modes maps all special values inf/nan to mx-scale nan
    shared_exponent = torch.where(
        max_exponent == dtypes.E8M0_EXPONENT_NAN_VAL,
        dtypes.E8M0_EXPONENT_NAN_VAL,
        shared_exponent,
    )

    return shared_exponent
