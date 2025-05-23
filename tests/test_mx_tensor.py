"""Tests adopted and modified from PyTorch AO repository.
https://github.com/pytorch/ao/blob/99c8d52c809cdb00b9f04e3657c9f3eae875a487/test/prototype/mx_formats/test_mx_tensor.py

"""

import pytest
import torch
import torch.nn.functional as F
from torchao.prototype.mx_formats.custom_cast import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
)
from torchao.quantization.utils import compute_error
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

from torchmx import dtypes
from torchmx.mx_tensor import MXTensor, dequantize_mx, quantize_mx
from torchmx.utils import pack_uint4, unpack_uint4

# trying to outsmart flake8
__has_cuda = torch.cuda.is_available()
IS_CUDA_GE_89 = __has_cuda and torch.cuda.get_device_capability() >= (8, 9)

torch.manual_seed(7)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # source: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test  # noqa: E501

    # setup (currently do nothing)

    # tests will run here
    yield

    # teardown
    # avoid dynamo cache limit issues
    torch._dynamo.reset()


def _test_mx(data_hp: torch.Tensor, elem_dtype: dtypes.DType, block_size: int):
    data_mx = MXTensor.to_mx(data_hp, elem_dtype, block_size)
    data_mx_dq = data_mx.to_dtype(data_hp.dtype)

    def assert_sqnr_gt_threshold(
        orig: torch.Tensor, new: torch.Tensor, threshold: float
    ):
        sqnr = compute_error(orig, new)
        if torch.all(torch.isnan(sqnr)):
            # if both operands are full of zeroes, sqnr is nan and this is ok
            # test for this explicitly
            assert torch.all(orig == 0) and torch.all(new == 0)
        else:
            assert sqnr >= threshold

    if elem_dtype is dtypes.float8_e4m3:
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 19.0)
    elif elem_dtype is dtypes.int8:
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 38.0)
    else:
        assert_sqnr_gt_threshold(data_hp, data_mx_dq, 14.0)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_hello_world(elem_dtype: dtypes.DType, device: str):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    data = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_all_zeros(elem_dtype: dtypes.DType, device: str):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    data = torch.zeros(4, 4, device=device, dtype=torch.bfloat16)
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_some_zeros(elem_dtype: dtypes.DType, device: str):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    data = torch.randn(4, 4, device=device, dtype=torch.bfloat16)
    data[0, :] = 0.0
    data[:, 2] = 0.0
    block_size = 2
    _test_mx(data, elem_dtype, block_size)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_exponent_nan_in(elem_dtype: dtypes.DType, device: str):
    """
    Test saturation mode, aka mx quantization when special cases such as inf/nan are present in the
    input tensor. In saturation mode the shared exponent is set to 255 and the elements are cast
    to a unsinged zero.
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    tensor_hp = torch.tensor(
        [float("nan"), -1, float("-inf"), float("inf")],
        device=device,
        dtype=torch.bfloat16,
    )
    tensor_lp_gt = (
        torch.zeros(2, device=device, dtype=torch.uint8)
        if elem_dtype == dtypes.float4_e2m1
        else torch.zeros(4, device=device, dtype=torch.uint8)
    )
    block_size = 4
    tensor_mx = MXTensor.to_mx(tensor_hp, elem_dtype, block_size)
    tensor_mx_dequant = tensor_mx.to_dtype(tensor_hp.dtype)
    assert torch.equal(
        tensor_mx._scale_e8m0,
        torch.tensor([dtypes.E8M0_EXPONENT_NAN_VAL], device=device, dtype=torch.uint8),
    )
    assert torch.equal(tensor_mx._data, tensor_lp_gt)
    assert torch.all(torch.isnan(tensor_mx_dequant))
    assert torch.all(torch.sign(tensor_mx_dequant) == 0)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_FP_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_special_values_input(
    elem_dtype: dtypes.DType,
    device: str,
    special_bfloat16_vector: torch.Tensor,
):
    """
    Test saturation mode, aka mx quantization when special cases such as inf/nan are present in the
    input tensor. In saturation mode the shared exponent is set to 255 and the elements are cast
    to a singed zero.
    """

    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    x = special_bfloat16_vector.to(device)
    shared_exponent, data_mx = quantize_mx(x, elem_dtype.name, 4)
    # The data must be signed zero
    gt_data = torch.zeros_like(x, dtype=torch.uint8)
    if elem_dtype == dtypes.float4_e2m1:
        gt_data = pack_uint4(gt_data)

    assert (shared_exponent == 255).all()
    assert torch.equal(data_mx, gt_data)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_ranks(elem_dtype: dtypes.DType, device: str):
    """
    The reshaping logic works for various ranks
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    B = 2
    shapes = ((B * 4,), (B * 4, 2), (B * 4, 2, 2), (B * 4, 2, 2, 2))
    for s in shapes:
        tensor_hp = torch.randn(*s, device=device, dtype=torch.bfloat16)
        _test_mx(tensor_hp, elem_dtype, B)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_block_sizes(elem_dtype: dtypes.DType, device: str):
    """
    Smoke test for various block sizes
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    for B in (1, 2, 32):
        if B == 1 and elem_dtype == dtypes.float4_e2m1:
            pytest.skip("unsupported configuration")
        tensor_hp = torch.randn(B, device=device, dtype=torch.bfloat16)
        _test_mx(tensor_hp, elem_dtype, B)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 25])
def test_t(elem_dtype: dtypes.DType | str, device: str, hp_dtype: torch.dtype, padding):
    """
    Verify the basic transpose operation tensor.t()
    This is same as tensor.transpose(0, 1)
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    tensor_hp = torch.randn(128, 256 - padding, device=device, dtype=torch.bfloat16)
    block_size = 32
    tensor_mx = MXTensor.to_mx(tensor_hp, elem_dtype, block_size)
    tensor_mx_dq_t = tensor_mx.to_dtype(tensor_hp.dtype).t()

    tensor_mx_t = tensor_mx.t()
    tensor_mx_t_dq = tensor_mx_t.to_dtype(tensor_hp.dtype)
    assert tensor_mx_dq_t.shape == tensor_mx_t_dq.shape
    torch.testing.assert_close(tensor_mx_dq_t, tensor_mx_t_dq, atol=0, rtol=0)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("padding", [0, 1])
def test_transpose(elem_dtype: dtypes.DType | str, device: str, padding: int):
    """
    Verify the basic transpose operation tensor.t(0,1)
    """
    HP_DTYPE = torch.bfloat16
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    tensor_hp = torch.randn(2, 32, 48, 4 + padding, device=device, dtype=HP_DTYPE)
    block_size = 2
    tensor_mx = MXTensor.to_mx(tensor_hp, elem_dtype, block_size)
    tensor_mx_dq_t = tensor_mx.to_dtype(tensor_hp.dtype).transpose(2, 3)

    tensor_mx_t = tensor_mx.transpose(2, 3)
    tensor_mx_t_dq = tensor_mx_t.to_dtype(tensor_hp.dtype)

    assert tensor_mx_dq_t.shape == tensor_mx_t_dq.shape
    torch.testing.assert_close(tensor_mx_dq_t, tensor_mx_t_dq, atol=0, rtol=0)


@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 3])
def test_cast_autograd(
    elem_dtype: dtypes.DType, device: str, hp_dtype: torch.dtype, padding: int
):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    x = torch.arange(8 + padding, device=device, dtype=torch.bfloat16).requires_grad_()
    grad = torch.arange(8 + padding, device=device, dtype=torch.bfloat16) * 0.5
    block_size = 8
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_dq = x_mx.to_dtype(hp_dtype)
    if elem_dtype == dtypes.float4_e2m1 and padding > 0:
        # Skip the test for float8 with padding
        with pytest.raises(ValueError):
            x_dq.backward(gradient=grad)
    else:
        x_dq.backward(gradient=grad)
        torch.testing.assert_close(grad, x.grad, atol=0, rtol=0)


@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("padding", [0, 12, 21])
def test_bmm(elem_dtype: dtypes.DType, device: str, padding: int):
    # This tests the spirit of attention matmul using 4D matmul as test tensor
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    torch_dtype = torch.float32 if device == "cpu" and padding > 0 else torch.bfloat16
    q = torch.randn(5, 2, 88, 128 + padding, device=device, dtype=torch.bfloat16)
    k = torch.randn(5, 2, 88, 128 + padding, device=device, dtype=torch.bfloat16)

    q_mx = MXTensor.to_mx(q, elem_dtype, 32)
    k_mx = MXTensor.to_mx(k, elem_dtype, 32)
    q_mx._orig_dtype = torch_dtype
    k_mx._orig_dtype = torch_dtype

    attn_weights = torch.matmul(q_mx, k_mx.transpose(2, 3))
    attn_weights_dq = torch.matmul(
        q_mx.to_dtype(torch_dtype), k_mx.to_dtype(torch_dtype).transpose(2, 3)
    )
    # Due to associatiy issues, the results may not be exactly equal when the accumulation
    # dimension is odd

    torch.testing.assert_close(attn_weights, attn_weights_dq, atol=0, rtol=0)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("padding", [0, 1])
def test_view(elem_dtype: dtypes.DType, device: str, padding: int):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    last_dim_size = 4 + padding
    x = torch.randn(11, 3, last_dim_size, device=device, dtype=torch.bfloat16)
    block_size = 2
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_dq = x_mx.to_dtype(torch.bfloat16)

    # # Apply view 1
    x_mx_2 = x_mx.view(11 * 3, last_dim_size)  # noqa: F841
    x_mx_2_dq = x_mx_2.to_dtype(torch.bfloat16)
    torch.testing.assert_close(x_mx_dq.view_as(x_mx_2_dq), x_mx_2_dq, atol=0, rtol=0)

    # Apply view 2 (flattening)
    def apply_flattening(last_dim_size, x_mx):
        x_mx_flattened = x_mx.view(11 * 3 * last_dim_size)  # noqa: F841
        return x_mx_flattened.to_dtype(torch.bfloat16)

    if padding > 0:
        # Check exception for flattening with padding
        with pytest.raises(AssertionError):
            apply_flattening(last_dim_size, x_mx)
    else:
        x_mx_3_dq = apply_flattening(last_dim_size, x_mx)
        torch.testing.assert_close(
            x_mx_dq.view_as(x_mx_3_dq), x_mx_3_dq, atol=0, rtol=0
        )

    # Check exception
    with pytest.raises(AssertionError):
        x_t = x_mx.transpose(1, 2)
        x_t.view(11 * 3 * 4)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("padding", [0, 1])
def test_view_4d(elem_dtype: dtypes.DType, device: str, padding: int):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    last_dim_size = 4 + padding
    x = torch.randn(5, 5, 3, last_dim_size, device=device, dtype=torch.bfloat16)
    block_size = 2
    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_dq = x_mx.to_dtype(torch.bfloat16)

    x_t = x_mx.transpose(2, 3)
    x_t_view = x_t.view(25, last_dim_size, 3)
    x_t_view_dq = x_t_view.to_dtype(torch.bfloat16)

    torch.testing.assert_close(
        x_t_view_dq, x_mx_dq.transpose(2, 3).view(25, last_dim_size, 3), atol=0, rtol=0
    )

    # Check exception
    x_t2 = x_mx.transpose(1, 3)
    with pytest.raises(AssertionError):
        # This will cause exception as we are squashing the block_dim
        x_t2.view(25, last_dim_size, 3)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_zeros", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("padding", [0, 1])
def test_to_mx_from_mx_compile_numerics(
    elem_dtype: dtypes.DType,
    hp_dtype: torch.dtype,
    all_zeros: bool,
    device: str,
    padding: int,
):
    """
    Verifies that compile does not change numerics of MX casts
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    if elem_dtype == dtypes.float8_e4m3:
        if not IS_CUDA_GE_89 and device == "cuda":
            pytest.skip("CUDA capability >= 8.9 required for float8 in triton")

    shape = 4, 8 + padding
    if not all_zeros:
        x = torch.randn(*shape, dtype=torch.bfloat16, device=device)
    else:
        x = torch.zeros(*shape, dtype=torch.bfloat16, device=device)
    block_size = 2
    to_mx_c = torch.compile(MXTensor.to_mx, fullgraph=True)

    x_mx = MXTensor.to_mx(x, elem_dtype, block_size)
    x_mx_c = to_mx_c(x, elem_dtype, block_size)
    torch.testing.assert_close(
        x_mx._scale_e8m0,
        x_mx_c._scale_e8m0,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(x_mx._data, x_mx_c._data, atol=0, rtol=0)

    to_dtype_c = torch.compile(dequantize_mx, fullgraph=True)

    x_mx_data = x_mx._data
    x_mx_c_data = x_mx_c._data
    if padding > 0 and elem_dtype != dtypes.float4_e2m1:
        x_mx_data = F.pad(x_mx_data, (0, padding), mode="constant", value=0.0)
        x_mx_c_data = F.pad(x_mx_c_data, (0, padding), mode="constant", value=0.0)

    x_mx_dq = dequantize_mx(
        x_mx_data,
        x_mx._scale_e8m0,
        str(x_mx._elem_dtype),
        x_mx._block_size,
        hp_dtype,  # noqa: E501
        x_mx._block_dim,
    )
    x_mx_c_dq = to_dtype_c(
        x_mx_c_data,
        x_mx_c._scale_e8m0,
        str(x_mx_c._elem_dtype),
        x_mx_c._block_size,
        hp_dtype,
        x_mx_c._block_dim,
    )
    torch.testing.assert_close(x_mx_dq, x_mx_c_dq, atol=0, rtol=0)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize(
    "input_shape", [(2, 4), (1, 4, 8), (1, 1, 8, 16), (2, 5), (1, 4, 9), (1, 1, 8, 17)]
)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_no_graph_breaks(
    elem_dtype: dtypes.DType,
    input_shape: tuple[int],
    device: str,
):
    """
    Verifies that MXTensors can be created without breaking the graph with torch.compile
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    def _mx_tensor_to_back(
        x: torch.Tensor, elem_dtype: dtypes.DType, block_size: int
    ) -> torch.Tensor:
        mx_tensor = MXTensor.to_mx(x, elem_dtype, block_size)
        return mx_tensor.to_dtype(x.dtype)

    block_size = 2
    x = torch.randn(*input_shape, dtype=torch.bfloat16, device=device)

    explanation = torch._dynamo.explain(_mx_tensor_to_back)(x, elem_dtype, block_size)
    assert (
        explanation.graph_break_count == 0
    ), f"Graph Breaks: {explanation.graph_break_count}"
    assert explanation.graph_count == 1, f"Graphs: {explanation.graph_count}"


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_quantize_mx_registered(
    elem_dtype: dtypes.DType,
    device: str,
):
    """
    Verifies that quantize_mx is registered with the torch.library
    this does not check the numerics of the function
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    shape = (4, 8)
    block_size = 2
    x = torch.randn(*shape, dtype=torch.bfloat16, device=device)

    args = (x, str(elem_dtype), block_size)
    result = torch.library.opcheck(quantize_mx, args)
    for k, v in result.items():
        assert v == "SUCCESS", f"Failed for {k}: {v}"


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("elem_dtype", dtypes.SUPPORTED_ELEM_DTYPES)
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("hp_dtype", [torch.float32, torch.bfloat16])
def test_dequantize_mx_registered(
    elem_dtype: dtypes.DType, device: str, hp_dtype: torch.dtype
):
    """
    Verifies that dequantize_mx is registered with the torch.library
    this does not check the numerics of the function
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    if elem_dtype == dtypes.float8_e4m3:
        if not IS_CUDA_GE_89 and device == "cuda":
            pytest.skip(
                "CUDA capability >= 8.9 required for float8. Else raises 'mul_cuda' not implemented for 'Float8_e4m3fn''"
            )
        elif device == "cpu":
            pytest.skip(
                "Skipping. Else causes RuntimeError: 'mul_cpu_reduced_float' not implemented for 'Float8_e4m3fn'"
            )
        else:
            raise ValueError(
                "Unsupported configuration. Ensure float8 and device compatibility."
            )

    shape = (4, 8)
    block_size = 2
    x = torch.randn(*shape, dtype=torch.bfloat16, device=device)
    mx_x = MXTensor.to_mx(x, elem_dtype, block_size)
    args = (
        mx_x._data,
        mx_x._scale_e8m0,
        str(elem_dtype),
        block_size,
        hp_dtype,
        mx_x._block_dim,
    )  # noqa: E501
    result = torch.library.opcheck(dequantize_mx, args)
    for k, v in result.items():
        assert v == "SUCCESS", f"Failed for {k}: {v}"


def test_fp4_pack_unpack():
    orig_vals = torch.Tensor([[0.0, 0.5, 4.0, -0.0], [-0.0, 1.0, -6.0, 3.0]])
    orig_vals_f4_unpacked = f32_to_f4_unpacked(orig_vals)
    orig_vals_f4_packed = pack_uint4(orig_vals_f4_unpacked)
    assert orig_vals_f4_packed.numel() == (orig_vals.numel() / 2)
    orig_vals_f4_packed_unpacked = unpack_uint4(orig_vals_f4_packed)
    orig_vals_dq = f4_unpacked_to_f32(orig_vals_f4_packed_unpacked)
    assert torch.all(orig_vals_dq == orig_vals)
