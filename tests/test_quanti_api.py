from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torchao.utils import TORCH_VERSION_AT_LEAST_2_4
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

from torchmx import quant_api
from torchmx.config import MXConfig, QAttentionConfig, QLinearConfig
from torchmx.layers.mx_linear import MXInferenceLinear
from torchmx.layers.mx_llama_attention import (
    MXInferenceLlamaAttention,
    MXInferenceLlamaMLP,
)
from torchmx.layers.mx_qwen2_attention import (
    MXInferenceQwen2Attention,
    MXInferenceQwen2MLP,
)

if not TORCH_VERSION_AT_LEAST_2_4:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


def test_quantize_linear():
    qconfig = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name="float6_e3m2",
            block_size=2,
        ),
        activations_config=MXConfig(
            elem_dtype_name="float8_e4m3",
            block_size=2,
        ),
    )
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),
        nn.Linear(in_features=16, out_features=20),
        nn.ReLU(),
        nn.Linear(in_features=20, out_features=50),
        nn.GELU(),
        nn.Linear(in_features=50, out_features=10),
    )
    model.to(torch.bfloat16)
    qmodel = deepcopy(model)
    quant_api.quantize_linear_(qmodel, qconfig)

    for layer, q_layer in zip(model.modules(), qmodel.modules()):
        if type(layer) == nn.Linear:
            assert type(q_layer) == MXInferenceLinear
        else:
            assert type(layer) == type(q_layer)


def test_quantize_linear_custom_forward():
    qconfig = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name="float6_e3m2",
            block_size=2,
        ),
        activations_config=MXConfig(
            elem_dtype_name="float8_e4m3",
            block_size=2,
        ),
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = Net().to(torch.bfloat16)
    qmodel = deepcopy(model)
    quant_api.quantize_linear_(qmodel, qconfig)

    for layer, q_layer in zip(model.modules(), qmodel.modules()):
        if type(layer) == nn.Linear:
            assert type(q_layer) == MXInferenceLinear
        else:
            assert type(layer) == type(q_layer)


def test_quantize_llm_():
    qattention_config = QAttentionConfig(
        projection_config=QLinearConfig(
            weights_config=MXConfig(
                elem_dtype_name="float6_e3m2",
                block_size=32,
            ),
            activations_config=MXConfig(
                elem_dtype_name="float8_e4m3",
                block_size=16,
            ),
        ),
        query_config=MXConfig(
            elem_dtype_name="float8_e4m3",
        ),
        key_config=MXConfig(
            elem_dtype_name="float4_e2m1",
        ),
        value_config=MXConfig(
            elem_dtype_name="float4_e2m1",
        ),
        attention_weights_config=MXConfig(
            elem_dtype_name="float6_e2m3",
        ),
    )

    qmlp_config = QLinearConfig(
        weights_config=MXConfig(
            elem_dtype_name="float6_e2m3",
            block_size=32,
        ),
        activations_config=MXConfig(
            elem_dtype_name="float8_e4m3",
            block_size=32,
        ),
    )

    llama_config = LlamaConfig(hidden_size=128, intermediate_size=128)
    qwen2config = Qwen2Config(hidden_size=128, intermediate_size=128)
    # The forward won't run because the shapes are not compatible.
    # This is just to see if the replacement works.
    model = nn.Sequential(
        nn.Linear(128, 128),
        LlamaAttention(llama_config),
        LlamaMLP(llama_config),
        nn.Softmax(dim=-1),
        nn.SiLU(),
        nn.Conv2d(3, 6, 5),
        Qwen2Attention(qwen2config),
        nn.SiLU(),
        Qwen2MLP(qwen2config),
    )
    model = model.to(torch.bfloat16)
    qmodel = deepcopy(model)
    quant_api.quantize_llm_(qmodel, qattention_config, qmlp_config)

    for layer, q_layer in zip(model.children(), qmodel.children()):
        if type(layer) == LlamaAttention:
            assert type(q_layer) == MXInferenceLlamaAttention, f"{type(q_layer)}"
        elif type(layer) == LlamaMLP:
            assert type(q_layer) == MXInferenceLlamaMLP, f"{type(q_layer)}"
        elif type(layer) == nn.Linear:
            assert type(q_layer) == MXInferenceLinear, f"{type(q_layer)}"
        elif type(layer) == Qwen2Attention:
            assert type(q_layer) == MXInferenceQwen2Attention, f"{type(q_layer)}"
        elif type(layer) == Qwen2MLP:
            assert type(q_layer) == MXInferenceQwen2MLP, f"{type(q_layer)}"
        else:
            assert type(layer) == type(q_layer)
