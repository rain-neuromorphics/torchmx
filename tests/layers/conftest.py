import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import torchmx.dtypes as dtypes

from tests.test_utils import set_seed

SIMULATED_LINEAR_SQNR = {
    "0": 41.5,
    "1": 19.25,
    "2": 41.5,
    "3": 19.25,
    "4": 41.5,
    "5": 19.25,
    "6": 41.5,
    "7": 19.25,
    "int8": 47.5,
}

SIMULATED_ATTEN_LINEAR_SQNR = {
    "0": 18,
    "1": 13,
    "2": 17,
    "3": 12,
    "4": 18,
    "5": 13,
    "6": 12,
    "7": 10,
}

SIMULATED_ATTEN_ALL_QUANT_SQNR = {
    "0": 17,
    "1": 11,
    "2": 16,
    "3": 12,
    "4": 17,
    "5": 12,
    "6": 12,
    "7": 8,
}

SIMULATED_MLP_SQNR = {
    "0": 16,
    "1": 9,
    "2": 14,
    "3": 8,
    "4": 16,
    "5": 9,
    "6": 10,
    "7": 7,
}


GEMM_COMBINATIONS = {
    "0": (dtypes.float8_e4m3, dtypes.float6_e3m2),
    "1": (dtypes.float8_e4m3, dtypes.float4_e2m1),
    "2": (dtypes.float6_e3m2, dtypes.float6_e3m2),
    "3": (dtypes.float6_e3m2, dtypes.float4_e2m1),
    "4": (dtypes.float6_e2m3, dtypes.float6_e3m2),
    "5": (dtypes.float6_e2m3, dtypes.float4_e2m1),
    "6": (dtypes.float4_e2m1, dtypes.float6_e3m2),
    "7": (dtypes.float4_e2m1, dtypes.float4_e2m1),
}


@pytest.fixture()
def weights_data():
    return torch.arange(4 * 6, dtype=torch.bfloat16).view(6, 4) + 0.123


@pytest.fixture()
def input_data():
    return torch.pow(2.0, -torch.arange(2 * 4).view(2, 4)).bfloat16()


@pytest.fixture()
def input_data_padded():
    return torch.pow(2.0, -torch.arange(2 * 6).view(2, 6)).bfloat16()


@pytest.fixture()
def hidden_states():
    # x = torch.load("tests/layers/hidden_states.pt")
    # return x\
    set_seed(42)
    return torch.rand(2, 128, 128, dtype=torch.bfloat16)


@pytest.fixture()
def llama_config():
    return LlamaConfig(
        hidden_size=128,
        num_key_value_heads=2,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
    )


@pytest.fixture()
def qwen2_config():
    return Qwen2Config(
        hidden_size=128,
        num_key_value_heads=2,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
    )


@pytest.fixture()
def simulated_linear_sqnr_gt():
    return SIMULATED_LINEAR_SQNR


@pytest.fixture()
def simulated_atten_linear_sqnr_gt():
    return SIMULATED_ATTEN_LINEAR_SQNR


@pytest.fixture()
def simulated_mlp_sqnr_gt():
    return SIMULATED_MLP_SQNR


@pytest.fixture()
def simulated_atten_all_quant_sqnr_gt():
    return SIMULATED_ATTEN_ALL_QUANT_SQNR
