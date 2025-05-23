import random

import numpy as np
import torch


def torch_equal_with_special(t1, t2):
    # Check if shapes are the same
    if t1.shape != t2.shape:
        return False

    # Check NaN consistency
    nan_mask1 = torch.isnan(t1)
    nan_mask2 = torch.isnan(t2)
    if not torch.equal(nan_mask1, nan_mask2):
        return False

    # Check +Inf and -Inf consistency
    inf_mask1 = torch.isinf(t1) & (t1 > 0)
    inf_mask2 = torch.isinf(t2) & (t2 > 0)
    if not torch.equal(inf_mask1, inf_mask2):
        return False

    ninf_mask1 = torch.isinf(t1) & (t1 < 0)
    ninf_mask2 = torch.isinf(t2) & (t2 < 0)
    if not torch.equal(ninf_mask1, ninf_mask2):
        return False

    # Check non-special values
    non_special_mask = ~(nan_mask1 | inf_mask1 | ninf_mask1)
    return torch.equal(t1[non_special_mask], t2[non_special_mask])


def sample_dtype_uniform(
    start: float, end: float, num_samples: int, dtype: str = "float32"
) -> torch.Tensor:
    """
    Generates a tensor of uniformly distributed random samples within a specified range and data type.

    Args:
        start (float): The lower bound of the range.
        end (float): The upper bound of the range.
        num_samples (int): The number of samples to generate.
        dtype (str, optional): The data type of the output tensor. Default is "float32".

    Returns:
        torch.Tensor: A tensor of uniformly distributed random samples.

    Raises:
        ValueError: If the start value is greater than the end value.
    """
    assert dtype in ["float32", "float64"], f"Unsupported dtype: {dtype}"
    if start > end:
        raise ValueError("Start value must be less than end value")
    return torch.rand(num_samples, dtype=getattr(torch, dtype)) * (end - start) + start


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
