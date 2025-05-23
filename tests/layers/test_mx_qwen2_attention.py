import pytest
import torch
from flaky import flaky
from torchao.quantization.utils import compute_error
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

from torchmx import dtypes
from torchmx.config import MXConfig, QAttentionConfig, QLinearConfig
from torchmx.layers.mx_qwen2_attention import (
    MXInferenceQwen2Attention,
    MXInferenceQwen2MLP,
)
from tests.layers.conftest import GEMM_COMBINATIONS
from tests.test_utils import set_seed

__has_cuda = torch.cuda.is_available()

# TODO: Add torch.compile tests with triton.


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    # source: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test  # noqa: E501

    # setup (currently do nothing)

    # tests will run here
    yield

    # teardown
    # avoid dynamo cache limit issues
    torch._dynamo.reset()


HIDDEN_SIZE = 128

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


def assert_sqnr_gt_threshold(orig: torch.Tensor, new: torch.Tensor, threshold: float):
    sqnr = compute_error(orig, new)
    if torch.all(torch.isnan(sqnr)):
        # if both operands are full of zeroes, sqnr is nan and this is ok
        # test for this explicitly
        assert torch.all(orig == 0) and torch.all(new == 0)
    else:
        assert sqnr >= threshold, f"sqnr: {sqnr}"


# TODO: Figure out why this test is flaky. Sometimes this results in all Nan in q_out.
# Rerunning the test with different random input passes.
# Note: both With and Without KV cache tests are flaky.
@flaky(max_runs=5, min_passes=1)
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mx_inference_qwen2_attention_without_kv_cache_quantization(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    device: str,
    hidden_states: torch.Tensor,
    qwen2_config: Qwen2Config,
    simulated_atten_linear_sqnr_gt: dict,
):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    set_seed(42)
    # Use a small hidden size for fast testing

    hidden_states = hidden_states.to(device)
    position_ids = torch.tensor(
        [[i for i in range(HIDDEN_SIZE)]], device=device, dtype=torch.int64
    )

    qwen2_layer = Qwen2Attention(qwen2_config).bfloat16().to(device)
    qwen2_layer.eval()
    hp_out = qwen2_layer(hidden_states, position_ids=position_ids)[0]

    q_qwen2_layer = MXInferenceQwen2Attention.from_float(
        mod=qwen2_layer,
        qconfig=QAttentionConfig(
            projection_config=QLinearConfig(
                weights_config=MXConfig(
                    elem_dtype_name=weight_dtype.name,
                    block_size=32,
                ),
                activations_config=MXConfig(
                    elem_dtype_name=input_dtype.name,
                    block_size=32,
                ),
            )
        ),
    )
    q_qwen2_layer.eval()
    q_out = q_qwen2_layer(hidden_states, position_ids=position_ids)[0]
    assert_sqnr_gt_threshold(hp_out, q_out, simulated_atten_linear_sqnr_gt[gemm_mode])


# Rerunning the test with different random input passes.
@flaky(max_runs=5, min_passes=1)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
def test_mx_inference_qwen2_attention_with_qkv_quantization(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    device: str,
    hidden_states: torch.Tensor,
    qwen2_config: Qwen2Config,
    simulated_atten_all_quant_sqnr_gt: dict,
):
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    set_seed(42)
    # Use a small hidden size for fast testing
    position_ids = torch.tensor(
        [[i for i in range(HIDDEN_SIZE)]], device=device, dtype=torch.int64
    )
    hidden_states = hidden_states.to(device)
    # Number of heads should be a multiple of maximum 4 to ensure that the block size is 32

    qwen2_layer = Qwen2Attention(qwen2_config).bfloat16().to(device)
    qwen2_layer.eval()
    hp_out = qwen2_layer(hidden_states, position_ids=position_ids)[0]

    qconfig = QAttentionConfig(
        projection_config=QLinearConfig(
            weights_config=MXConfig(
                elem_dtype_name=weight_dtype.name,
                block_size=32,
            ),
            activations_config=MXConfig(
                elem_dtype_name=input_dtype.name,
                block_size=32,
            ),
        ),
        query_config=MXConfig(
            elem_dtype_name=input_dtype.name,
            block_size=32,
        ),
        key_config=MXConfig(
            elem_dtype_name=weight_dtype.name,
            block_size=32,
        ),
        value_config=MXConfig(
            elem_dtype_name=weight_dtype.name,
            block_size=32,
        ),
        attention_weights_config=MXConfig(
            elem_dtype_name=input_dtype.name,
            block_size=32,
        ),
    )

    q_qwen2_layer = MXInferenceQwen2Attention.from_float(
        mod=qwen2_layer,
        qconfig=qconfig,
    )
    q_qwen2_layer.eval()
    q_out = q_qwen2_layer(hidden_states, position_ids=position_ids)[0]
    assert_sqnr_gt_threshold(
        hp_out, q_out, simulated_atten_all_quant_sqnr_gt[gemm_mode]
    )


@pytest.mark.parametrize(
    "gemm_mode,input_dtype,weight_dtype",
    [(k, v[0], v[1]) for k, (v) in GEMM_COMBINATIONS.items()],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mx_inference_qwen2_mlp(
    gemm_mode: str,
    input_dtype: dtypes.DType,
    weight_dtype: dtypes.DType,
    device: str,
    simulated_mlp_sqnr_gt: dict,
    hidden_states: torch.Tensor,
    qwen2_config: Qwen2Config,
):
    set_seed(42)
    if device == "cuda" and not __has_cuda:
        pytest.skip("CUDA not available")
    hidden_states = hidden_states.to(device)
    # Number of heads should be a multiple of maximum 4 to ensure that the block size is 32
    qwen2_mlp = Qwen2MLP(qwen2_config).bfloat16().to(device)
    qwen2_mlp.eval()
    hp_out = qwen2_mlp(hidden_states)

    q_qwen2_mlp = MXInferenceQwen2MLP.from_float(
        mod=qwen2_mlp,
        qconfig=QLinearConfig(
            weights_config=MXConfig(
                elem_dtype_name=weight_dtype.name,
                block_size=32,
            ),
            activations_config=MXConfig(
                elem_dtype_name=input_dtype.name,
                block_size=32,
            ),
        ),
    )
    q_qwen2_mlp.eval()
    q_out = q_qwen2_mlp(hidden_states)
    assert_sqnr_gt_threshold(hp_out, q_out, simulated_mlp_sqnr_gt[gemm_mode])
