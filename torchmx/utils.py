import logging
import math
from typing import Iterable, List
import random

import numpy as np
import torch

from . import env_variables as env_v


def get_logger(
    logger_name: str = "TORCHMX",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    console_output: bool = True,
) -> logging.Logger:
    """Returns a logger with the specified name and format.

    Args:
        logger_name (str, optional): Name of the logger. Defaults to "TORCHMX".
        format_string (str, optional): Format of the log message. Defaults to "%(asctime)s - %(name)s - %(levelname)s - %(message)s".
        console_output (bool, optional): Whether to output the logs to the console. Defaults to True.

    Returns:
        logging.Logger: Logger with the specified name and format.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(env_v.TORCHMX_LOG_LEVEL)
    formatter = logging.Formatter(format_string)
    if console_output:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if env_v.TORCHMX_LOG_FILE:
        file_handler = logging.FileHandler(env_v.TORCHMX_LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # If not, logger sometimes prints multiple times.
    logger.propagate = False
    return logger


def get_uniform_random_number(
    min_val: int, max_val: int, shape: Iterable[int], dtype: torch.dtype
) -> torch.Tensor:
    """Generate random numbers from uniform distribution of range [min_val, max_val)

    Args:
        min_val (int): minimum value of the range
        max_val (int): maximum value of the range
        shape (Iterable[int]): shape of the tensor
        dtype (torch.dtype): data type of the tensor

    Returns:
        torch.Tensor: tensor of shape `shape` and dtype `dtype` with random numbers
    """
    return (max_val - min_val) * torch.rand(*shape, dtype=dtype) + min_val


def tensor_size_hp_to_fp4x2(
    orig_size: torch.Size,
    packing_dim: int,
) -> List[int]:
    """Converts the size of a tensor from half precision to fp4x2 precision.

    Args:
        orig_size (torch.Size): The size of the original tensor.
        packing_dim (int): The dimension where for packing 2xuint4 per byte
    Returns:
        List[int]: The size of the tensor in fp4x2 precision.
    """
    new_size = list(orig_size)
    new_size[packing_dim] = math.ceil(new_size[packing_dim] / 2)
    return new_size


def tensor_size_fp4x2_to_hp(
    orig_size: torch.Size,
    unpacking_dim: int,
) -> List[int]:
    """Converts the size of a tensor from fp4x2 precision to half precision by unpacking the
    4-bits into 8-bits.

    Args:
        orig_size (torch.Size): The size of the original tensor.
        unpacking_dim (int): The dimension where for unpacking the uint4 values to a single byte
    Returns:
        List[int]: The size of the tensor in half precision.
    """
    new_size = list(orig_size)
    new_size[unpacking_dim] = new_size[unpacking_dim] * 2
    return new_size


def unpack_uint4(uint8_data: torch.Tensor, packing_dim: int = -1) -> torch.Tensor:
    """
    Unpacks a tensor of uint8 values into two tensors of uint4 values.

    Args:
        uint8_data (torch.Tensor): A tensor containing packed uint8 values.
        packing_dim (int): The dimension along which the unpacking is performed.

    Returns:
        torch.Tensor: A tensor containing the unpacked uint4 values.
    """
    if packing_dim < 0:
        packing_dim += uint8_data.dim()
    shape = uint8_data.shape
    up_size = tensor_size_fp4x2_to_hp(shape, packing_dim)
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)

    unpacked = torch.stack([first_elements, second_elements], dim=packing_dim + 1).view(
        up_size
    )
    return unpacked


def pack_uint4(uint8_data: torch.Tensor, packing_dim: int = -1) -> torch.Tensor:
    """
    Packs uint4 data to unit8 format along the specified dimension.

    Args:
        uint4_data (torch.Tensor): The input tensor containing uint8 data.
        packing_dim (int): The dimension along which to pack the data.

    Returns:
        torch.Tensor: A tensor with the packed uint4 data.

    Raises:
        AssertionError: If the size of the specified dimension is not even.

    Note:
        The function assumes that the input data is contiguous and reshapes it
        accordingly. The packing is done by combining pairs of uint8 values into
        a single uint8 value where each original uint8 value is treated as a uint4.
    """
    # converting to uint8 for operations
    shape = uint8_data.shape
    down_size = tensor_size_hp_to_fp4x2(shape, packing_dim)
    # TODO: We need to be able to pack even for odd sizes
    assert shape[packing_dim] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size)


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
