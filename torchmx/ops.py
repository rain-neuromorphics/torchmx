"""All the custom ops for the TorchMX package are defined here.
This also includes the aten ops that are implemented for the MXTensor class.
"""

from copy import deepcopy

import torch
from torch.utils._pytree import tree_map

from . import dtypes
from .mx_tensor import MXTensor
from .utils import get_logger, tensor_size_hp_to_fp4x2

logger = get_logger(__name__)

# Force set bf16 and fp16 to not use reduced precision reduction for GEMMs
# https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-reduction-for-fp16-and-bf16-gemms
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


############################### Aten Ops Implementation ###############################

aten = torch.ops.aten

implements = MXTensor.implements


@implements([aten.linear.default])
def mx_linear(aten_op, types, args, kwargs=None):
    a = args[0]
    b = args[1]
    if len(args) > 2:
        c = args[2]
    else:
        c = None
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)
    res = aten_op(a_hp, b_hp, c)
    return res


@implements([aten.detach.default])
def mx_desugar_op(aten_op, types, args, kwargs=None):
    old = args[0]
    new_data = aten_op(old._data, *args[1:], **kwargs)
    new = MXTensor(
        old._scale_e8m0,
        new_data,
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._padding,
        old._block_dim,
    )
    return new


@implements([aten.mm.default, aten.matmul.default])
def mx_mm(aten_op, types, args, kwargs=None):
    a = args[0]
    b = args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)
    res = aten_op(a_hp, b_hp)
    return res


@implements([aten.expand.default])
def mx_expand(aten_op, types, args, kwargs=None):
    # This is called before calling the bmm op in case of 4D matmul.
    # We infer the scale_expand_size from the data_expand_size which is given by args[1]
    # The scale_expand_size is the same as the data_expand_size but the block_dim is
    # divided by the block size. This only implements the minimum expand op required to
    # support the 4D matmul in attention.
    old = args[0]
    scale_expand_size = deepcopy(args[1])
    scale_expand_size[old._block_dim] = (
        scale_expand_size[old._block_dim] + old._padding
    ) // old._block_size
    data_expand_size = args[1]
    if args[0]._elem_dtype == dtypes.float4_e2m1:
        # Special case fp4 as we pack two elements per byte
        data_expand_size = tensor_size_hp_to_fp4x2(data_expand_size, old._block_dim)
    new = MXTensor(
        aten_op(old._scale_e8m0, scale_expand_size, *args[2:], **kwargs),
        aten_op(old._data, data_expand_size, *args[2:], **kwargs),
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._padding,
        old._block_dim,
    )
    return new


@implements([aten.bmm.default])
def mx_bmm(aten_op, types, args, kwargs=None):
    a = args[0]
    b = args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)
    res = aten_op(a_hp, b_hp)
    return res


@implements([aten.addmm.default])
def mx_addmm(aten_op, types, args, kwargs=None):
    a = args[0]
    b = args[1]
    c = args[2]
    assert isinstance(b, MXTensor) and isinstance(c, MXTensor)
    b_hp = b.to_dtype(b._orig_dtype)
    c_hp = c.to_dtype(c._orig_dtype)
    res = aten_op(a, b_hp, c_hp)
    return res


@implements([aten.t.default])
def mx_t(aten_op, types, args, kwargs=None):
    old = args[0]
    assert old._block_dim in [0, 1]
    new_block_dim = 1 if old._block_dim == 0 else 0
    new = MXTensor(
        old._scale_e8m0.t(),
        old._data.t(),
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._padding,
        new_block_dim,
    )
    return new


@implements([aten.transpose.int])
def mx_transpose(aten_op, types, args, kwargs=None):
    # For supporting operations like tensor.transpose(2,3)
    old = args[0]
    # if the block_dim is in the transposed dims, then swap the block_dim
    # otherwise the block_dim remains the same
    if old._block_dim in args[1:]:
        new_block_dim = args[1] if old._block_dim == args[2] else args[2]
    else:
        new_block_dim = old._block_dim
    new = MXTensor(
        aten_op(old._scale_e8m0, *args[1:], **kwargs),
        aten_op(old._data, *args[1:], **kwargs),
        old._elem_dtype,
        old._block_size,
        old._orig_dtype,
        old._padding,
        new_block_dim,
    )
    return new


@implements([aten.sum.dim_IntList])
def mx_cast_up_op(aten_op, types, args, kwargs=None):
    """Be careful with this function, this is a "fallback" op that
    casts the output of the op to the original precision. And performs the op.

    We currently need this to support the backward for admmm bias.
    "addmm" -> out
    "hp_gradBias" <-"sum" <- "identity" <- gradOut <- "hp_gradOut"
    """

    def unwrap(x):
        if isinstance(x, MXTensor):
            return x.to_dtype(x._orig_dtype)
        return x

    new_args = tree_map(unwrap, args)
    new_kwargs = tree_map(unwrap, kwargs)
    return aten_op(*new_args, **new_kwargs)


@implements([aten.view.default])
def mx_view_op(aten_op, types, args, kwargs=None):
    """
    This is a custom op that is used to handle the view op for MXTensor. The user
    is not expected to call this op directly. This Op is only implemented to support
    some internal PyTorch functions. We only supports view op in the case following
    cases:
        - When the block dim is the last dim
            - This is needed for aten.linear
        - When the block dim is the second last dim:
            - The tensor must be 4D, else raises Assertion error
            - This is needed for the following 4D matmuls in attention:
                - torch.matmul(query_states, key_states.transpose(2, 3))
                - torch.matmul(attn_weights, value_states)

    Raises:
        In all the other cases we raise ValueError
    """
    data = args[0]._data
    new_size = args[1]
    scale_new_size = new_size.copy()

    # If the tensor is padded, we need to add the padding to the new size
    if len(scale_new_size) == 1 and args[0]._padding > 0:
        raise AssertionError(
            "View op is not supported when the tensor is padded and the new size is 1D"
        )

    # Convert the block_dim to negative
    negative_block_dim = args[0]._block_dim - data.ndim

    # We only support this operation when the block dim is last or second last dim
    assert (
        negative_block_dim >= -2
    ), "View Op is supported only when block_dim is last/second_last dim"
    # To support attention matmul in 4D: torch.matmul(attn_weights, value_states)
    # Block dim is the second last dim
    if negative_block_dim == -2:
        assert (
            data.ndim == 4
        ), "For the view op when block_dim is second last dim, the tensor must be 4D"

    # Calculate the block dim for the new shape
    new_block_dim = len(new_size) + negative_block_dim
    # Calculate the new size for the scale tensor
    scale_new_size[negative_block_dim] = (
        scale_new_size[negative_block_dim] + args[0]._padding
    ) // args[0]._block_size

    if args[0]._elem_dtype == dtypes.float4_e2m1:
        # special case fp4 as we pack two elements per byte
        new_size = tensor_size_hp_to_fp4x2(new_size, new_block_dim)
    # Transform the data to the new size
    new_data = aten_op(data, new_size, *args[2:], **kwargs)

    # Transform the scale to the new size
    scale_e8m0_view = aten_op(args[0]._scale_e8m0, scale_new_size, *args[2:], **kwargs)

    res = MXTensor(
        scale_e8m0_view,
        new_data,
        args[0]._elem_dtype,
        args[0]._block_size,
        args[0]._orig_dtype,
        args[0]._padding,
        new_block_dim,
    )
    return res


@implements([aten._to_copy.default])
def autocast_to_copy(aten_op, types, args, kwargs=None):
    """This gets called when running matmul under autocast
    when the input is a MXTensor, presenting as a fp32
    tensor.
    """
    assert isinstance(args[0], MXTensor)
    # print('before', args[0], args[0].dtype, args[0]._orig_dtype)
    assert (
        len(kwargs) == 1 and "dtype" in kwargs
    ), "Only support dtype kwarg for autocast"
    assert kwargs["dtype"] in {
        torch.float16,
        torch.bfloat16,
    }, "Only support floating point conversion for autocast w/ MXTensor"
    res = MXTensor(
        args[0]._scale_e8m0,
        args[0]._data,
        args[0]._elem_dtype,
        args[0]._block_size,
        kwargs["dtype"],
        args[0]._padding,
        args[0]._block_dim,
    )
    # print('after', res, res.dtype, res._orig_dtype)
    return res
