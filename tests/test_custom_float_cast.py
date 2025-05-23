from functools import partial

import pytest
import torch

from torchmx import dtypes
from torchmx.custom_float_cast import hp_to_floatx
from tests.test_utils import set_seed
from tests.test_utils import sample_dtype_uniform, torch_equal_with_special

set_seed(42)

hp_to_float8_e4m3 = partial(
    hp_to_floatx,
    exponent_bits=dtypes.float8_e4m3.exponent_bits,
    mantissa_bits=dtypes.float8_e4m3.mantissa_bits,
    max_normal=dtypes.float8_e4m3.max,
)
hp_to_float22_e8m13 = partial(
    hp_to_floatx,
    exponent_bits=dtypes.float22_e8m13.exponent_bits,
    mantissa_bits=dtypes.float22_e8m13.mantissa_bits,
    max_normal=dtypes.float22_e8m13.max,
)


class TestFloat8E4M3:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_all_float8_e4m3_values(self, dtype: str):
        """
        all fp8_e4m3 values -> cast to fp32 -> convert back to fp8_e4m3 -> compare with
        fp32 input
        """
        torch_dtype = getattr(torch, dtype)
        # Generate of all possible values of a float8
        float8_uint = torch.arange(0, 256, dtype=torch.uint8)
        float8_hp = float8_uint.view(dtypes.float8_e4m3.torch_dtype).to(torch_dtype)

        y = hp_to_float8_e4m3(float8_hp)
        assert torch_equal_with_special(float8_hp, y)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_uniform_sample(self, dtype: str):
        """
        Take uniform FP32 sample from a range 2x the max value of float8_e4m3 -> cast to float8_e4m3
        -> compare with direct casting using torch.float8_e4m3_fn (clamped to max value)
        This test implicitly tests correct round to even behavior and saturation.
        """
        # This implicitly tests the round_to_nearest
        torch_dtype = getattr(torch, dtype)
        fp_vector = sample_dtype_uniform(
            -dtypes.float8_e4m3.max * 2.0, dtypes.float8_e4m3.max * 2.0, 10000, dtype
        )
        y = hp_to_float8_e4m3(fp_vector)
        y_gt = (
            fp_vector.clamp_(-dtypes.float8_e4m3.max, dtypes.float8_e4m3.max)
            .to(dtypes.float8_e4m3.torch_dtype)
            .to(torch_dtype)
        )
        assert torch_equal_with_special(y_gt, y)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_saturation(self, dtype: str):
        """
        Test saturation behavior for float8_e4m3 and handling  of special values
        Note: the special values handling is not consistent with OCP , as input infs are
        mapped to infs, even if float8_e4m3 does not support infs. This function is designed to
        simulate custom float conversion only. For exact behaviour, resort to `quantize_mx`
        """
        torch_dtype = getattr(torch, dtype)
        max_normal = torch.finfo(torch_dtype).max
        fp_vector = torch.tensor(
            [
                float("nan"),
                float("-inf"),
                -max_normal,
                -500,
                500,
                +max_normal,
                float("inf"),
                float("-nan"),
            ],
            dtype=torch_dtype,
        )
        y = hp_to_float8_e4m3(fp_vector)
        y_gt = torch.tensor(
            [
                float("nan"),
                float("-inf"),
                -dtypes.float8_e4m3.max,
                -dtypes.float8_e4m3.max,
                dtypes.float8_e4m3.max,
                dtypes.float8_e4m3.max,
                float("inf"),
                float("nan"),
            ],
            dtype=torch_dtype,
        )
        assert torch_equal_with_special(y, y_gt)


class TestFloat22E8M13:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_all_float22_values(
        self, dtype: str, all_float22_e8m13_values: torch.Tensor
    ):
        """
        all float22_e8m13 values -> cast to fp32 -> convert back to float22_e8m13 -> compare with
        fp32 input
        """
        torch_dtype = getattr(torch, dtype)
        x = all_float22_e8m13_values.to(torch_dtype)
        y = hp_to_floatx(
            x,
            dtypes.float22_e8m13.exponent_bits,
            dtypes.float22_e8m13.mantissa_bits,
            dtypes.float22_e8m13.max,
        )
        assert torch_equal_with_special(x, y)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_all_float22_values_no_subnormals(
        self, dtype: str, all_float22_e8m13_values: torch.Tensor
    ):
        """
        all float22_e8m13 values -> cast to fp32 -> convert back to float22_e8m13 -> compare with
        fp32 input
        """
        torch_dtype = getattr(torch, dtype)
        x = all_float22_e8m13_values.to(torch_dtype)
        y = hp_to_floatx(
            x,
            dtypes.float22_e8m13.exponent_bits,
            dtypes.float22_e8m13.mantissa_bits,
            dtypes.float22_e8m13.max,
            keep_subnormals=False,
        )
        min_norm_fp22 = 2**-126
        x = torch.where(torch.abs(x) < min_norm_fp22, torch.zeros_like(x), x)
        assert torch_equal_with_special(x, y)

    def test_round_to_even_fp32_values(self):
        """
        Construct float32 values and convert them to float22 using hp_to_float22_e8m13
        Then we compare with FP22 ground truth assuming "round_to_even" rounding mode.
        Note: float32 and float22 have the same exponent
        """
        fp32_exponent = torch.tensor([0, 127, 50, 230], dtype=torch.int32)
        fp32_mantissa = torch.tensor(
            [
                0b1001000100000_1100100010,
                0b1011000100001_0100100010,
                0b1001000100001_1100100010,
                0b0001110100000_0100100010,
            ],
            dtype=torch.int32,
        )
        fp32_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        int32_input = (fp32_sign << 31) | (fp32_exponent << 23) | fp32_mantissa
        fp32_input = int32_input.view(torch.float32)

        # Construct ground truth
        fp22_mantissa = torch.tensor(
            [
                0b1001000100001,
                0b1011000100001,
                0b1001000100010,
                0b0001110100000,
            ],
            dtype=torch.int32,
        )
        fp22_as_uint32 = (fp32_sign << 31) | (fp32_exponent << 23) | fp22_mantissa << 10
        fp22_output = fp22_as_uint32.view(torch.float32)

        y = hp_to_float22_e8m13(fp32_input)
        assert torch_equal_with_special(fp22_output, y)

    def test_truncation_fp32_values(self):
        """
        Construct float32 values and convert them to float22 using hp_to_float22_e8m13
        Then we compare with FP22 ground truth assuming "truncate" rounding mode.
        Note: float32 and float22 have the same exponent
        """
        fp32_exponent = torch.tensor([0, 127, 50, 230], dtype=torch.int32)
        fp32_mantissa = torch.tensor(
            [
                0b1001000100000_1100100010,
                0b1011000100001_0100100010,
                0b1001000100001_1100100010,
                0b0001110100000_0100100010,
            ],
            dtype=torch.int32,
        )
        fp32_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        int32_input = (fp32_sign << 31) | (fp32_exponent << 23) | fp32_mantissa
        fp32_input = int32_input.view(torch.float32)
        fp22_mantissa = fp32_mantissa >> 10

        fp22_as_int32 = (fp32_sign << 31) | (fp32_exponent << 23) | fp22_mantissa << 10
        fp22_output = fp22_as_int32.view(torch.float32)

        y = hp_to_float22_e8m13(fp32_input, round_mode="truncate")
        assert torch_equal_with_special(fp22_output, y)

    def test_round_to_even_fp64_values(self):
        """
        Construct float64 values and convert them to float22 using hp_to_float22_e8m13
        Then we compare with FP22 ground truth assuming "round_to_even" rounding mode. \
        As float64 and float22 have different exponent we cover cases:
            1. normal -> underflow
            2. normal -> overflow
            3. normal -> subnormal
            4. normal -> normal
        """
        fp64_exponent = torch.tensor([100, 888, 1000, 2000], dtype=torch.int64)
        fp64_mantissa = torch.tensor(
            [
                0b1001000100000_1100100010,
                0b1011100100001_0100100010,
                0b1001000100001_1100100010,
                0b0001110100000_0100100010,
            ],
            dtype=torch.int64,
        )
        fp64_mantissa = fp64_mantissa << 29
        fp64_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        int64_input = (fp64_sign << 63) | (fp64_exponent << 52) | fp64_mantissa
        fp64_input = int64_input.view(torch.float64)

        # Construct ground truth
        fp22_exponent = torch.tensor([0, 0, 104, 254], dtype=torch.int32)
        fp22_mantissa = torch.tensor(
            [
                0,
                0b11100,
                0b1001000100010,
                0b1111111111111,
            ],
            dtype=torch.int32,
        )
        fp22_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        fp22_as_uint32 = (fp22_sign << 31) | (fp22_exponent << 23) | fp22_mantissa << 10
        fp22_output = fp22_as_uint32.view(torch.float32).to(torch.float64)

        y = hp_to_float22_e8m13(fp64_input)
        assert torch_equal_with_special(fp22_output, y)

    def test_truncate_fp64_values(self):
        """
        Construct float64 values and convert them to float22 using hp_to_float22_e8m13
        Then we compare with FP22 ground truth assuming "truncate" rounding mode. \
        As float64 and float22 have different exponent we cover cases:
            1. normal -> underflow
            2. normal -> overflow
            3. normal -> subnormal
            4. normal -> normal
        """
        fp64_exponent = torch.tensor([100, 888, 1000, 2000], dtype=torch.int64)
        fp64_mantissa = torch.tensor(
            [
                0b1001000100000_1100100010,
                0b1011100100001_0100100010,
                0b1001000100001_1100100010,
                0b0001110100000_0100100010,
            ],
            dtype=torch.int64,
        )
        fp64_mantissa = fp64_mantissa << 29
        fp64_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        int64_input = (fp64_sign << 63) | (fp64_exponent << 52) | fp64_mantissa
        fp64_input = int64_input.view(torch.float64)

        # Construct ground truth
        fp22_exponent = torch.tensor([0, 0, 104, 254], dtype=torch.int32)
        fp22_mantissa = torch.tensor(
            [
                0,
                0b11011,
                0b1001000100001,
                0b1111111111111,
            ],
            dtype=torch.int32,
        )
        fp22_sign = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        fp22_as_uint32 = (fp22_sign << 31) | (fp22_exponent << 23) | fp22_mantissa << 10
        fp22_output = fp22_as_uint32.view(torch.float32).to(torch.float64)

        y = hp_to_float22_e8m13(fp64_input, round_mode="truncate")
        assert torch_equal_with_special(fp22_output, y)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_saturation(self, dtype: str):
        """
        Test saturation behavior for float22_e8m13 and handling  of special values
        """
        torch_dtype = getattr(torch, dtype)
        max_normal = torch.finfo(torch_dtype).max
        fp_vector = torch.tensor(
            [
                float("nan"),
                float("-inf"),
                -max_normal,
                max_normal,
                float("inf"),
                float("-nan"),
            ],
            dtype=torch_dtype,
        )
        y = hp_to_float22_e8m13(fp_vector)
        y_gt = torch.tensor(
            [
                float("nan"),
                float("-inf"),
                -dtypes.float22_e8m13.max,
                dtypes.float22_e8m13.max,
                float("inf"),
                float("nan"),
            ],
            dtype=torch_dtype,
        )
        assert torch_equal_with_special(y, y_gt)
