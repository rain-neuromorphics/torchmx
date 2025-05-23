import torch

# code adapted from  https://github.com/microsoft/microxcaling/blob/main/mx/elemwise_ops.py


def _get_min_norm(exponent_bits: int) -> float:
    """
    Calculate the minimum normal for a floating-point format given the number of exponents.

    Args:
        exponent_bits (int): The number of bits used for the exponent in the floating-point representation.

    Returns:
        float: The minimum normalized floating-point number. Returns 0 if exponent_bits is 0.
    """
    emin = 2 - (2 ** (exponent_bits - 1))
    return 0 if exponent_bits == 0 else 2**emin


def hp_to_floatx(
    hp_data: torch.Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    max_normal: float,
    round_mode="round_to_even",
    keep_subnormals: bool = True,
):
    """
    Converts high-precision floating-point data to a custom floating-point format, as specified by
    the number of exponent and mantissa bits.
    Notes:
        - This function does not take into account whether the data format supportes NaNs or infs.
          It will return the NaNs and Infs as found in the input tesnor
        - It implemention  OCP's 'saturating mode' the values to the max_normal value.

    Args:
        hp_data (torch.Tensor): Input tensor with high-precision floating-point data (float32 or float64).
        exponent_bits (int): Number of bits for the exponent in the target format
        mantissa_bits (int): Number of bits for the mantissa in the target format.
        max_normal (float): Maximum representable normal value in the target format.
        round_mode (str, optional): Rounding mode to use. Options are "truncate" and
        "round_to_even". Default is "round_to_even".
        keep_subnormals (bool, optional): Whether to keep subnormal values. Default is True.

    Returns:
        torch.Tensor: Tensor with data converted to the custom floating-point format.
    """
    assert hp_data.dtype in [
        torch.float64,
        torch.float32,
    ], f"Invalid data type: {hp_data.dtype}"
    assert round_mode in [
        "truncate",
        "round_to_even",
    ], f"Invalid round mode: {round_mode}"

    # Flush values < min_norm to zero if subnormals are not allowed
    if not keep_subnormals:
        min_norm = _get_min_norm(exponent_bits)
        hp_data = torch.where(
            torch.abs(hp_data) < min_norm, torch.zeros_like(hp_data), hp_data
        )

    # Get the biased exponent of the input
    unbiased_exponent = torch.floor(
        torch.log2(torch.abs(hp_data) + (hp_data == 0).type(hp_data.dtype))
    )

    # min representable exponent for the target format
    target_min_exponent = -(2 ** (exponent_bits - 1)) + 2
    unbiased_exponent = unbiased_exponent.clip(min=target_min_exponent)

    # Scale up so appropriate number of bits are in the integer portion of the number
    mantissa = hp_data / (2**unbiased_exponent) * (2**mantissa_bits)

    rounded_mantissa = _round_mantissa(mantissa, round_mode)

    # Undo scaling
    out = rounded_mantissa / (2**mantissa_bits) * (2**unbiased_exponent)

    # Clamp values larger than max_normal
    saturation_mask = (
        torch.abs(hp_data) > max_normal
    )  # this deals with very large values in the input
    out = torch.where(saturation_mask, torch.sign(hp_data) * max_normal, out)

    # Maintain infs
    out = torch.where(hp_data.isinf(), hp_data, out)

    return out


def _round_mantissa(
    mantissa: torch.Tensor, round_mode: str = "nearest"
) -> torch.Tensor:
    """
    Rounds the mantissa of a tensor according to the specified rounding mode

    Args:
        mantissa (torch.Tensor): The input tensor containing mantissa values to be rounded.
        round_mode (str, optional): The rounding mode to use. Can be "truncate", "round_to_even", or "nearest". Defaults to "nearest".

    Returns:
        torch.Tensor: The rounded mantissa tensor.

    Raises:
        ValueError: If an invalid rounding mode is specified.
    """
    if round_mode == "truncate":
        mantissa = torch.sign(mantissa) * torch.floor(torch.abs(mantissa))
    elif round_mode == "round_to_even":
        abs_mantissa = torch.abs(mantissa)
        # find 0.5, 2.5, 4.5 ...
        even_mask = ((abs_mantissa - 0.5) % 2 == torch.zeros_like(mantissa)).type(
            mantissa.dtype
        )
        mantissa = torch.sign(mantissa) * (torch.floor(abs_mantissa + 0.5) - even_mask)
    else:
        raise ValueError(f"Invalid rounding mode: {round_mode}")

    return mantissa
