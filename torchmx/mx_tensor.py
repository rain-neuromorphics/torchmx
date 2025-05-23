# Adapted with modifications from https://github.com/pytorch/ao/blob/v0.6.1/torchao/prototype/mx_formats/mx_tensor.py

"""
Defines the tensor subclasses to represent the [OCP MX-Format spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

Exponent E8M0 encoding details (OCP spec section 5.4.1):

  * bias: 127
  * supported exponent range: -127 to 127
  * infinities: N/A
  * NaN: 11111111
  * Zeros: N/A
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5, TorchAOBaseTensor

from . import dtypes
from . import env_variables as env
from .mx_quantization_utils import (
    dequantize_to_dtype,
    get_e8m0_shared_exponent,
    get_fp_scale,
    quantize_mx_with_e8m0_shared_exponent_hw_exact,
    quantize_mx_with_e8m0_shared_exponent_simulated,
)
from .utils import get_logger, pack_uint4, tensor_size_fp4x2_to_hp, unpack_uint4

logger = get_logger(__name__)


@torch.library.custom_op("torchmx::quantize_mx", mutates_args=())
def quantize_mx(
    data_hp: torch.Tensor,
    elem_dtype_name: str,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Takes a high precision tensor and converts to MX scale and raw data, in
    naive layout (scale and raw data are separate tensors). The function for now only
    supports quantization along the last dimension of the input tensor. For example,
    if the input tensor has shape (N, C, H, W) the output will be:
        - data_lp (torch.uint8) with shape (N, C, H, W)
        - scale (torch.uint8) with shape (N, C, H, W // block_size)

    Args:
        data_hp (torch.Tensor): high precision data tensor (dtype=torch.bfloat16)
        elem_dtype_name (str): target element dtype as a string to comply with torch.library.infer_schema
        block_size (int): block size

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scale(biased), low precision data as tensors
    """
    elem_dtype = dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[elem_dtype_name]
    assert (
        data_hp.dtype == torch.bfloat16
    ), f"Only torch.bfloat16 input dtype is supported, got {data_hp.dtype}"
    assert data_hp.is_contiguous(), "unsupported"
    assert (
        elem_dtype in dtypes.SUPPORTED_ELEM_DTYPES
    ), f"Unsupported dtype {elem_dtype}. Supported: {dtypes.SUPPORTED_ELEM_DTYPES}"

    # TODO(future PR): consider supporting padding
    assert (
        data_hp.shape[-1] % block_size == 0
    ), "The last dimension of the input tensor must be a multiple of block_size"
    # calculate the scale in e8m0 format, by first reshaping the input tensor.
    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # get the shared exponent for the input tensor as a torch.uint8 tensor
    shared_exponent = get_e8m0_shared_exponent(data_hp, elem_dtype)

    # normalize the input tensor by the shared exponent and cast to the target elem_dtype
    # if hardware_exact_implementation is True, use the exact MX quantization
    if (
        elem_dtype in dtypes.SUPPORTED_FP_ELEM_DTYPES
        and env.MX_EXACT_QUANTIZATION == "True"
    ):
        data_lp = quantize_mx_with_e8m0_shared_exponent_hw_exact(
            data_hp, elem_dtype, shared_exponent.unsqueeze(1), orig_shape
        )
    else:
        data_lp = quantize_mx_with_e8m0_shared_exponent_simulated(
            data_hp, elem_dtype, shared_exponent.unsqueeze(1), orig_shape
        )
    # Reshape the shared exponent to match the input tensor
    output_shape = orig_shape[:-1] + (-1,)
    shared_exponent = shared_exponent.reshape(
        output_shape
    )  # the last dim the number of blocks
    return shared_exponent, data_lp


@quantize_mx.register_fake
def _(
    data_hp: torch.Tensor, elem_dtype_name: str, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fake quantize_mx implementation.

    This adds a “FakeTensor kernel” (aka “meta kernel”) to the operator. Given some
    FakeTensors inputs (dummy Tensors that don't have storage), this function returns
    dummy Tensors with the correct Tensor metadata(shape/strides/dtype/device). This is
    used by torch.compile to infer the shape and other metadata of the output tensors.

    Reference: https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial
    """
    elem_dtype = dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[elem_dtype_name]
    fake_lp = torch.empty_like(data_hp)
    orig_shape = data_hp.shape
    scale_shape = orig_shape[:-1] + (orig_shape[-1] // block_size,)
    scale = torch.empty(size=scale_shape, dtype=torch.uint8, device=data_hp.device)
    fake_lp = fake_lp.to(torch.uint8 if elem_dtype != dtypes.int8 else torch.int8)
    if elem_dtype == dtypes.float4_e2m1:
        fake_lp = pack_uint4(fake_lp)
    return scale.to(torch.uint8), fake_lp


@torch.library.custom_op("torchmx::dequantize_mx", mutates_args=())
def dequantize_mx(
    data_lp: torch.Tensor,
    shared_exp_e8m0: torch.Tensor,
    elem_dtype_name: str,
    block_size: int,
    target_dtype: torch.dtype,
    block_dim: int,
) -> torch.Tensor:
    """Takes the low precision data and scale of MXTensor and converts to high precision

    Args:
        data_lp (torch.Tensor): low precision data tensor
        shared_exp_e8m0 (torch.Tensor): biased exponent of the shared MX scale as torch.uint8
        elem_dtype_name (str): target element dtype's name as a string to comply with torch.library.infer_schema
        block_size (int): block size
        target_dtype (torch.dtype): target dtype
        block_dim (int): block dimension

    Returns:
        torch.Tensor: high precision data tensor in target_dtype converted from MX
    """
    elem_dtype = dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[elem_dtype_name]
    # TODO: rewrite the triton kernels to support the block_dim

    if elem_dtype in dtypes.SUPPORTED_FP_ELEM_DTYPES:
        # dequantize the data_lp to the target_dtype by first casting to bfloa16
        data_hp = dequantize_to_dtype(data_lp, elem_dtype, target_dtype, block_dim)

    elif elem_dtype == dtypes.int8:
        data_hp = data_lp.to(target_dtype)
    else:
        raise AssertionError(f"unsupported dtype: {elem_dtype}")

    s_fp = (
        get_fp_scale(shared_exp_e8m0)
        .to(target_dtype)
        .repeat_interleave(block_size, dim=block_dim)
    )
    data_hp = data_hp * s_fp

    return data_hp


@dequantize_mx.register_fake
def _(
    data_lp: torch.Tensor,
    shared_exp_e8m0: torch.Tensor,
    elem_dtype_name: str,
    block_size: int,
    target_dtype: torch.dtype,
    block_dim: int,
) -> torch.Tensor:
    """Fake dequantize_mx implementation.

    This adds a “FakeTensor kernel” (aka “meta kernel”) to the operator. Given some
    FakeTensors inputs (dummy Tensors that don't have storage), this function returns
    dummy Tensors with the correct Tensor metadata(shape/strides/dtype/device). This is
    used by torch.compile to infer the shape and other metadata of the output tensors.

    Reference: https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial
    """
    elem_dtype = dtypes.STR_TO_SUPPORTED_ELEM_DTYPE[elem_dtype_name]
    data_lp = torch.empty_like(data_lp)

    if elem_dtype == dtypes.float4_e2m1:
        data_lp = unpack_uint4(data_lp, block_dim)

    data_hp = data_lp.to(target_dtype)

    return data_hp


@torch._dynamo.allow_in_graph
class ToMXConstrFunc(torch.autograd.Function):
    """
    Differentiable cast to MX, no-op in backward
    """

    @staticmethod
    def forward(ctx, data_hp: torch.Tensor, elem_dtype: dtypes.DType, block_size: int):
        """
        Forward method for the custom autograd function.
        Args:
            ctx: The context object that can be used to stash information
                 for backward computation.
            data_hp (torch.Tensor): The high-precision input tensor to be quantized.
            elem_dtype (dtypes.DType): The target data type for quantization.
            block_size (int): The block size used for quantization.
            padding (int): The padding size applied during quantization.
        Returns:
            MXTensor: A custom tensor object containing the quantized data,
                      scale factor, and metadata about the quantization process.
        """
        # pad the input tensor to be a multiple of block_size
        padding = (block_size - data_hp.shape[-1] % block_size) % block_size
        size_before_padding = data_hp.shape[-1]
        if padding > 0:
            data_hp = F.pad(data_hp, (0, padding))  # Only pad the last dimension

        # Apply mx quantization
        scale_e8m0_biased, data_lp = quantize_mx(data_hp, elem_dtype.name, block_size)

        # Reshape the mx data to the original shape
        if padding > 0:
            assert (
                block_size % 2 == 0
            ), f"block_size must be even to support padding but got {block_size}"
            if elem_dtype == dtypes.float4_e2m1:
                # Need to account for the 2x4 packing for float4_e2m1. We use ceil to
                # ensure we capture the all elements. For example, assuming block_size = 4
                # and input data x = [1, 2, 3, 4, 5], the padded data will be
                # data_lp will be have length 4. The 5th element will be packed into the 3rd block

                size_before_padding = math.ceil(
                    size_before_padding / 2
                )  # we use ceil to en

        return MXTensor(
            scale_e8m0_biased,
            data_lp[..., :size_before_padding].contiguous(),
            elem_dtype,
            block_size,
            data_hp.dtype,
            padding,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None


@torch._dynamo.allow_in_graph
class FromMXConstrFunc(torch.autograd.Function):
    """
    Differentiable cast from MX, no-op in backward
    """

    @staticmethod
    def forward(
        ctx, tensor_lp: torch.Tensor, target_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Forward method for dequantizing a low-precision tensor to a target data type.
        Args:
            ctx: The context object (not used in this implementation).
            tensor_lp (torch.Tensor): The low-precision tensor to be dequantized.
                This tensor is expected to have the following attributes:
                    - _data: The raw data of the tensor.
                    - _scale_e8m0: The scale factor for dequantization.
                    - _elem_dtype.name: The name of the element data type.
                    - _block_size: The block size used in the tensor.
                    - _block_dim: The block dimensions of the tensor.
                    - _padding: The amount of padding applied to the tensor.
            target_dtype (torch.dtype): The target data type to which the tensor
                should be dequantized.
        Returns:
            torch.Tensor: The dequantized tensor, reshaped to its original shape
            if padding was involved.
        """

        # Find the original size of the tensor before padding
        ctx._padding = tensor_lp._padding
        ctx._elem_dtype = tensor_lp._elem_dtype
        data_lp = tensor_lp._data
        org_size = tensor_lp._data.shape[tensor_lp._block_dim]
        if tensor_lp._elem_dtype == dtypes.float4_e2m1:
            # Need to account for the 2x4 packing for float4_e2m1.
            # if the paddig is not 0, we need to adjust the org_size to remove the zero that
            # had been packed into the last value
            org_size = org_size * 2 - (tensor_lp._padding % 2)

        # Pad the low precision tensor if required
        if tensor_lp._padding > 0:
            # Pad the tensor to perform dequantization
            padding = [0] * (data_lp.ndim * 2)
            padding_size = tensor_lp._padding
            # Adjust the padding size for the 4x2 packing
            if tensor_lp._elem_dtype == dtypes.float4_e2m1:
                padding_size = padding_size // 2

            padding[data_lp.ndim * 2 - tensor_lp._block_dim * 2 - 1] = padding_size
            data_lp = F.pad(data_lp, padding, mode="constant", value=0.0)

        # Apply dequantization
        mx_dequant = dequantize_mx(
            data_lp,
            tensor_lp._scale_e8m0,
            tensor_lp._elem_dtype.name,
            tensor_lp._block_size,
            target_dtype,
            tensor_lp._block_dim,
        )

        # Undo the padding
        if tensor_lp._padding > 0:
            slicer = [slice(None)] * tensor_lp.dim()
            slicer[tensor_lp._block_dim] = slice(0, org_size)
            mx_dequant = mx_dequant[tuple(slicer)]

        return mx_dequant.contiguous()

    @staticmethod
    def backward(ctx, g):
        if ctx._padding > 0 and ctx._elem_dtype == dtypes.float4_e2m1:
            raise ValueError(
                "Padding is not supported in the backward pass for float4_e2m1"
            )
        return g, None, None


@torch._dynamo.allow_in_graph
class NoopFwToMXBw(torch.autograd.Function):
    """
    Forward: no-op
    Backward: cast grad to MX
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, elem_dtype: dtypes.DType, block_size: int):
        ctx.elem_dtype = elem_dtype
        ctx.block_size = block_size
        return x

    @staticmethod
    def backward(ctx, g):
        scale, data = quantize_mx(g, ctx.elem_dtype.name, ctx.block_size)
        return (
            MXTensor(scale, data, ctx.elem_dtype, ctx.block_size, g.dtype),
            None,
            None,
        )


class MXTensor(TorchAOBaseTensor):
    def __new__(
        cls,
        scale_e8m0_bits: torch.Tensor,
        data_bits: torch.Tensor,
        elem_dtype: dtypes.DType,
        block_size: int,
        orig_dtype: torch.dtype,
        padding: int = 0,
        block_dim: Optional[int] = None,
    ):
        """
        Create a new instance of the tensor subclass.
        Args:
            cls: The class being instantiated.
            scale_e8m0_bits (torch.Tensor): A tensor containing scale factors with dtype torch.uint8.
            data_bits (torch.Tensor): A tensor containing data bits.
            elem_dtype (dtypes.DType): The element data type.
            block_size (int): The block size.
            orig_dtype (torch.dtype): The original data type.
            block_dim (int): The block dimension. Default is None. If not set it defaults to the
            last dimension.
            padding (int): Padding size in case the block_dim is not multiple of the block_size
            Default is 0.
        Returns:
            An instance of the tensor subclass.
        Raises:
            AssertionError: If the dtype of scale_e8m0_bits is not torch.uint8.
            AssertionError: If the shape of scale_e8m0_bits is not 1-dimensional.
            AssertionError: If the dtype of data_bits is not one of the supported types.
            AssertionError: If elem_dtype is unsupported.
        """
        # if th block_dim is not set, default to the last dimension
        if block_dim is None:
            block_dim = data_bits.dim() - 1
        else:
            # Ensure the block_dim is positive
            block_dim = block_dim if block_dim >= 0 else block_dim + data_bits.dim()

        new_size = data_bits.size()
        if elem_dtype == dtypes.float4_e2m1:
            # set the tensor size to what it would be without 2x4 packing
            new_size = tensor_size_fp4x2_to_hp(new_size, block_dim)
            # removes the additional element that was added to the last dimension
            # to account for the 2x4 packing with odd padding
            new_size[block_dim] -= padding % 2
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            new_size,
            dtype=orig_dtype,
            device=data_bits.device,
        )
        assert scale_e8m0_bits.dtype == torch.uint8, "unsupported"
        assert (
            elem_dtype in dtypes.SUPPORTED_ELEM_DTYPES
        ), f"unsupported elem_dtype {elem_dtype}"
        assert data_bits.dtype in (
            torch.uint8,
            torch.int8,
        ), f"{data_bits.dtype} is unsupported"
        # Get the expected number of elements according the scale_e8m0_bits
        scale_e8m0_bits_shape = list(scale_e8m0_bits.shape)
        scale_e8m0_bits_shape[block_dim] = (
            scale_e8m0_bits_shape[block_dim] * block_size - padding
        )
        target_numel = torch.Size(scale_e8m0_bits_shape).numel()
        if not issubclass(
            torch._subclasses.fake_tensor.FakeTensor,
            type(data_bits),
        ):
            # Caclulate the number of elements in the data_bits tensor
            data_bits_shape = list(data_bits.shape)
            if elem_dtype == dtypes.float4_e2m1:
                # Need to account for the 2x4 packing for float4_e2m1.
                data_bits_shape[block_dim] = data_bits_shape[block_dim] * 2 - (
                    padding % 2
                )
            data_numel = torch.Size(data_bits_shape).numel()
            assert target_numel == data_numel, f"{target_numel} != {data_numel}"

        self._scale_e8m0 = scale_e8m0_bits
        self._data = data_bits
        self._elem_dtype = elem_dtype
        self._block_size = block_size
        self._orig_dtype = orig_dtype
        self._block_dim = block_dim
        self._padding = padding
        return self

    def _quantization_type(self):
        # For printing purposes
        return f"shape={self.shape}, block_size={self._block_size}, device={self.device}, elem_dtype={self._elem_dtype}, orig_dtype={self._orig_dtype}, "

    def __repr__(self):
        repr_string = f"MXTensor: _elem_dtype: {self._elem_dtype}, _scale_e8m0: {self._scale_e8m0}, _data: {self._data}, d_hp: {self.to_dtype(self._orig_dtype)}"  # noqa: E501
        if self._padding > 0:
            repr_string += f", padding: {self._padding}"
        return repr_string

    def to_dtype(self, target_dtype: torch.dtype) -> torch.Tensor:
        """Dequantize the MXTensor to the target_dtype.

        Args:
            target_dtype (torch.dtype): The target data type \
                (torch.bfloat16, torch.float32, etc.) \
                to which the MXTensor is dequantized.

        Returns:
            The dequantized tensor in the target_dtype.

        Note:
            The MXTensor quantization is supported only for `torch.bfloat16`. But we \
            allow the dequantization to either `torch.bfloat16` or `torch.float32`.
            Look at the `quantize_mx` and `de_quantize_mx` functions for more details.
        """
        return FromMXConstrFunc.apply(self, target_dtype)

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp: torch.Tensor,
        elem_dtype: dtypes.DType,
        block_size: int = 32,
    ) -> "MXTensor":
        """Convert/Quantize a high-precision tensor to MXTensor.

        Args:
            data_hp (torch.Tensor): The high-precision input tensor. \
                  Only `torch.bfloat16` is supported. Look at the `quantize_mx` \
                  function for more details.
            elem_dtype (dtypes.DType): The target element data type for quantization.
            block_size (int): The block size. Default is 32.

        Returns:
            MXTensor: The quantized tensor in the target lower precision format.
        """
        return ToMXConstrFunc.apply(data_hp, elem_dtype, block_size)

    def __tensor_flatten__(self):
        ctx = {
            "_elem_dtype": self._elem_dtype,
            "_block_size": self._block_size,
            "_orig_dtype": self._orig_dtype,
            "_block_dim": self._block_dim,
            "_padding": self._padding,
        }
        return ["_scale_e8m0", "_data"], ctx

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict,
        metadata,
        outer_size,
        outer_stride,
    ):
        return MXTensor(
            inner_tensors["_scale_e8m0"],
            inner_tensors["_data"],
            metadata["_elem_dtype"],
            metadata["_block_size"],
            metadata["_orig_dtype"],
            metadata["_padding"],
            metadata["_block_dim"],
        )

    # Do not force the MXTensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl


if TORCH_VERSION_AT_LEAST_2_5:
    # Allow a model with MXTensor weights to be loaded with `weights_only=True`
    torch.serialization.add_safe_globals([MXTensor])
