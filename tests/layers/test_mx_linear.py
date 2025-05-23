# Adopted with modifications from: https://github.com/pytorch/ao/blob/1a0dbf1c41ad1c6f28d6501e1134b30ea2f2590d/test/prototype/mx_formats/test_mx_linear.py

import copy
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchao.quantization.utils import compute_error
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4

from torchmx import dtypes
from torchmx.config import MXConfig, QLinearConfig
from torchmx.quant_api import quantize_linear_
from tests.layers.conftest import GEMM_COMBINATIONS

__has_cuda = torch.cuda.is_available()

IS_CUDA_GE_89 = __has_cuda and torch.cuda.get_device_capability() >= (
    8,
    9,
)


def swap_linear_with_mx_inference_linear(
    m: nn.Module, elem_dtype: dtypes.DType, block_size: int, act_dtype: dtypes.DType
):
    """
    Swap linear layer with MXInferenceLinear
    """
    qconfig = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name=elem_dtype.name,
            block_size=block_size,
        ),
        activations_config=MXConfig(
            elem_dtype_name=act_dtype.name, block_size=block_size
        ),
    )
    quantize_linear_(m, qconfig=qconfig)


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # source: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test  # noqa: E501

    # setup (currently do nothing)

    # tests will run here
    yield

    # teardown
    # avoid dynamo cache limit issues
    torch._dynamo.reset()


torch.manual_seed(2)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("input_shape", [(2, 4), (1, 2, 4), (1, 1, 2, 4)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_inference_linear(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    input_shape: Tuple[int],
    device: str,
    inference_mode: bool,
    weights_data: torch.Tensor,
    input_data: torch.Tensor,
    simulated_linear_sqnr_gt: dict,
):
    """
    Smoke test for inference linear module with mx weight
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    # Create the model
    m = nn.Sequential(nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)).to(device)
    m[0].weight.data = weights_data.to(device).clone()
    m_mx = copy.deepcopy(m)

    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, weight_dtype, block_size, input_dtype)

    # Create the input data
    x = input_data.view(*input_shape).to(device)
    x.requires_grad = not inference_mode

    if not inference_mode:
        y_ref = m(x)
        y_mx = m_mx(x)
    else:
        # Note: When using F.linear():
        # - aten.linear gets called only when using torch.inference_mode().
        # - Otherwise, aten.mm is called when bias is disabled and aten.addmm is called when bias is enabled
        with torch.inference_mode():
            y_ref = m(x)
            y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx).item()
    assert np.isclose(
        sqnr, simulated_linear_sqnr_gt[gemm_mode], atol=0
    ), f"sqnr: {sqnr}"


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(4, 4), (1, 4, 4), (1, 1, 4, 4)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inference_mode", [True])
def test_special_values_inference(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    bias: bool,
    input_shape: Tuple[int],
    device: str,
    inference_mode: bool,
):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    m = nn.Sequential(nn.Linear(4, 6, bias=bias, dtype=torch.bfloat16)).to(device)
    m_mx = copy.deepcopy(m)
    block_size = 4
    swap_linear_with_mx_inference_linear(m_mx, weight_dtype, block_size, input_dtype)
    x = torch.tensor(
        [
            [float("inf"), 0, 100, 200],
            [float("-inf"), 0, 100, 200],
            [float("nan"), 0, 100, 200],
            [float("nan"), 0, float("-inf"), float("inf")],
        ],
        device=device,
        dtype=torch.bfloat16,
        requires_grad=not inference_mode,
    ).view(*input_shape)
    if not inference_mode:
        y_ref = m(x)
        y_mx = m_mx(x)
    else:
        # Note: When using F.linear():
        # - aten.linear gets called only when using torch.inference_mode().
        # - Otherwise, aten.mm is called when bias is disabled and aten.addmm is called when bias is enabled
        with torch.inference_mode():
            y_ref = m(x)
            y_mx = m_mx(x)
    y_gt = (
        torch.tensor(
            [torch.nan],
            dtype=torch.bfloat16,
            device=device,
            requires_grad=not inference_mode,
        )
        .repeat(4, 6)
        .view_as(y_ref)
    )
    assert torch.all(
        torch.isnan(y_mx) == torch.isnan(y_gt)
    ), f"y_mx: {y_mx}, y_gt: {y_gt}"


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("input_shape", [(2, 4), (1, 2, 4), (1, 1, 2, 4)])
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inference_mode", [True])
def test_inference_compile_simple(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    input_shape: Tuple[int],
    device: str,
    inference_mode: bool,
    weights_data: torch.Tensor,
    input_data: torch.Tensor,
    simulated_linear_sqnr_gt: dict,
):
    """
    Smoke test for inference compile
    """

    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    # Create the model
    m = nn.Sequential(nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)).to(device)
    m[0].weight.data = weights_data.to(device).clone()
    m_mx = copy.deepcopy(m)

    block_size = 2
    swap_linear_with_mx_inference_linear(m_mx, weight_dtype, block_size, input_dtype)
    m_mx = torch.compile(m_mx, fullgraph="true")

    # Create the input data
    x = input_data.view(*input_shape).to(device)
    x.requires_grad = not inference_mode

    if not inference_mode:
        y_ref = m(x)
        y_mx = m_mx(x)
    else:
        # Note: When using F.linear():
        # - aten.linear gets called only when using torch.inference_mode().
        # - Otherwise, aten.mm is called when bias is disabled and aten.addmm is called when bias is enabled
        with torch.inference_mode():
            y_ref = m(x)
            y_mx = m_mx(x)
    sqnr = compute_error(y_ref, y_mx).item()
    assert np.isclose(
        sqnr, simulated_linear_sqnr_gt[gemm_mode], atol=0
    ), f"sqnr: {sqnr}"


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize("input_shape", [(2, 6), (1, 2, 6), (1, 1, 2, 6)])
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("inference_mode", [True])
def test_inference_compile_padded(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    input_shape: Tuple[int],
    device: str,
    inference_mode: bool,
    weights_data: torch.Tensor,
    input_data_padded: torch.Tensor,
):
    """
    Smoke test for inference compile with padding
    """

    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    # Create the model
    m = nn.Sequential(nn.Linear(6, 4, bias=False, dtype=torch.bfloat16)).to(device)
    m[0].weight.data = weights_data.t().contiguous().to(device).clone()
    m_mx = copy.deepcopy(m)

    block_size = 4
    swap_linear_with_mx_inference_linear(m_mx, weight_dtype, block_size, input_dtype)
    m_mx = torch.compile(m_mx, fullgraph="true")

    # Create the input data
    x = input_data_padded.view(*input_shape).to(device)
    x.requires_grad = not inference_mode

    if not inference_mode:
        _ = m_mx(x)
    else:
        # Note: When using F.linear():
        # - aten.linear gets called only when using torch.inference_mode().
        # - Otherwise, aten.mm is called when bias is disabled and aten.addmm is called when bias is enabled
        with torch.inference_mode():
            _ = m_mx(x)


@pytest.mark.usefixtures("set_quantization_env")
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("padding", [0, 1])
def test_no_graph_break_inference_linear(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    device: str,
    padding: int,
):
    """
    Test that inference linear does not break graph
    """
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")

    m_mx = nn.Sequential(
        nn.Linear(4 + padding, 6, bias=False, dtype=torch.bfloat16)
    ).to(device)
    block_size = 2
    x = torch.randn(2, 4 + padding, dtype=torch.bfloat16, device=device)
    swap_linear_with_mx_inference_linear(m_mx, weight_dtype, block_size, input_dtype)
    explanation = torch._dynamo.explain(m_mx)(x)
    assert (
        explanation.graph_break_count == 0
    ), f"Graph Breaks: {explanation.graph_break_count}"
    assert explanation.graph_count == 1, f"Graphs: {explanation.graph_count}"
