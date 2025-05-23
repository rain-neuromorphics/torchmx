import pytest
import torch

from torchmx import dtypes
from torchmx import env_variables as env


@pytest.fixture()
def bfloat16_all_normals():
    mantissa = torch.arange(128).to(torch.int16).view(1, 1, -1)
    exponent = (torch.arange(1, 255).to(torch.int16) << 7).view(1, -1, 1)
    sign = (torch.Tensor([0, 1]).to(torch.int16) << 15).view(-1, 1, 1)

    bfloat16_ints = (sign | exponent | mantissa).flatten()

    bfloat16_tensor = bfloat16_ints.view(torch.bfloat16)
    return bfloat16_tensor.msort()


@pytest.fixture()
def bloat16_subnormals():
    mantissa = torch.arange(1, 128).to(torch.int16).view(1, 1, -1)
    sign = (torch.Tensor([0, 1]).to(torch.int16) << 15).view(-1, 1, 1)

    bfloat16_ints = (sign | mantissa).flatten()

    bfloat16_tensor = bfloat16_ints.view(torch.bfloat16)
    return bfloat16_tensor.msort()


@pytest.fixture()
def all_bfloat16_values(bfloat16_all_normals, bloat16_subnormals):
    return torch.cat([bfloat16_all_normals, bloat16_subnormals]).msort()


@pytest.fixture()
def all_float22_e8m13_values():
    sign_shift = dtypes.float32.mantissa_bits + dtypes.float32.exponent_bits  # 31
    exponent_shift = dtypes.float32.mantissa_bits  # 23
    mantissa_shift = dtypes.float32.mantissa_bits - dtypes.float22_e8m13.mantissa_bits
    mantissa = torch.arange(0, 2**13).to(torch.int32).view(1, 1, -1) << mantissa_shift
    exponent = (torch.arange(0, 255).to(torch.int32) << exponent_shift).view(1, -1, 1)
    sign = (torch.Tensor([0, 1]).to(torch.int32) << sign_shift).view(-1, 1, 1)

    float22_e8m13_ints = (sign | exponent | mantissa).flatten()

    float22_e8m13_tensor = float22_e8m13_ints.view(torch.float32)
    special_cases = torch.tensor(
        [float("nan"), float("-inf"), float("inf"), float("-nan")], dtype=torch.float32
    )
    output = torch.cat([float22_e8m13_tensor, special_cases])
    return output.msort()


@pytest.fixture()
def special_bfloat16_vector():
    x = torch.randn(5, 4, dtype=torch.bfloat16)
    x[0, 1] = float("inf")
    x[1, 1] = float("-inf")
    x[2, 1] = float("nan")
    x[3, 1] = -float("nan")
    x[4, 1], x[4, 2] = float("nan"), float("inf")
    return x


@pytest.fixture(params=["True", "False"])
def set_quantization_env(request):
    env.MX_EXACT_QUANTIZATION = request.param
    yield
