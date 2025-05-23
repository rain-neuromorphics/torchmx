import pytest
import torch
from torchao.prototype.mx_formats.custom_cast import unpack_uint4

from torchmx import dtypes
from torchmx import env_variables as env
from torchmx.mx_quantization_utils import round_to_even
from torchmx.mx_tensor import MXTensor


class TestMXFloat8e4m3:
    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_normal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0001010, 0b1000001, 0b1, 0b0101010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor(
            [
                [5, 5, 5, 5, 5, 19],
                [100, 100, 100, 100, 100, 111],
                [240, 240, 240, 240, 240, 249],
            ]
        ).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        scale_e8m0_gt = torch.tensor([11, 103, 241]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [0b0, 0b001, 0b100, 0b0, 0b011, 0], dtype=torch.uint8
        ).view(1, -1)
        gt_exponent = torch.tensor(
            [[2, 1, 1, 1, 1, 15], [5, 4, 4, 4, 4, 15], [7, 6, 6, 6, 6, 15]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 7 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, scale_e8m0_gt.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_saturation(self):
        # Construct the input
        bf16_exponent = torch.tensor([100, 100, 100]).to(torch.int16)
        bf16_mantissa = torch.tensor(
            [0b1110010, 0b1110010, 0b1111110], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 1], dtype=torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([92]).to(torch.uint8)
        mx_scale = 2.0 ** (shared_exponent.bfloat16() - 127)
        gt = mx_scale * torch.tensor(
            [-dtypes.float8_e4m3.max, dtypes.float8_e4m3.max, -dtypes.float8_e4m3.max],
            dtype=torch.bfloat16,
        )
        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 3)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, gt)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_subnormal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0001010, 0b1000001, 0b1, 0b0101010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor([100]).to(torch.int16).repeat(3, 6)
        bf16_exponent[0, -1], bf16_exponent[1, -1], bf16_exponent[2, -1] = 118, 116, 115
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        gt_mantissa = torch.tensor(
            [
                [0b1, 0b1, 0b1, 0b1, 0b1, 0],
                [0b100, 0b010, 0b011, 0b010, 0b011, 0],
                [0b0, 0b100, 0b110, 0b100, 0b101, 0],
            ]
        ).to(torch.uint8)

        gt_exponent = torch.tensor(
            [[0, 0, 0, 0, 0, 15], [0, 0, 0, 0, 0, 15], [1, 0, 0, 0, 0, 15]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 7 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)

        shared_exponent = torch.tensor([110, 108, 107]).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_underflow(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0001010, 0b1000001, 0b1, 0b0101010, 0], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.int16)
        bf16_exponent = torch.tensor([100, 100, 100, 100, 100, 119]).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 6)
        y = y_mx.to_dtype(torch.bfloat16)
        # Construct the ground truth
        y_gt = torch.tensor(
            [float("-0"), 0, 0, float("-0"), 0, 2**-8], dtype=torch.bfloat16
        )
        assert torch.equal(y, y_gt)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_zeros_to_zeros(self):
        # Construct the input
        bfloat16_inps = torch.zeros(3, 6, dtype=torch.bfloat16)
        bfloat16_inps[0, -1], bfloat16_inps[1, -1], bfloat16_inps[2, -1] = (
            2**-9,
            2**5,
            2**-125,
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 6)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, bfloat16_inps)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_subnormals_to_normal_and_subnormal(self):
        # Construct the input
        bf16_mantissa = (
            torch.tensor(
                [0b1111111, 0b0001010, 0b1000001, 0b0110011, 0b0101010, 0],
                dtype=torch.int16,
            )
            .view(1, -1)
            .expand(3, -1)
        )
        bf16_sign = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.zeros(3, 6).to(torch.int16)
        bf16_exponent[0, -1], bf16_exponent[1, -1], bf16_exponent[2, -1] = 12, 13, 14
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([4, 5, 6]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [
                [0b0, 0b101, 0b000, 0b101, 0b010, 0],
                [0b0, 0b10, 0b0, 0b101, 0b010, 0],
                [0b0, 0b1, 0b0, 0b110, 0b101, 0],
            ],
            dtype=torch.uint8,
        )

        gt_exponent = torch.tensor(
            [[4, 0, 3, 2, 2, 15], [3, 0, 2, 1, 1, 15], [2, 0, 1, 0, 0, 15]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 7 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float8_e4m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
    def test_random_bfloat16_combinations_hw_exact_and_simulated(
        self, all_bfloat16_values: torch.Tensor, target_dtype: torch.dtype
    ):
        # Generate a random permutation of row indices
        row_indices = torch.randperm(all_bfloat16_values.shape[0])[:-30]
        x = all_bfloat16_values[row_indices]

        # Hardware exact
        env.MX_EXACT_QUANTIZATION = "True"
        y_hw_exact = MXTensor.to_mx(x, dtypes.float8_e4m3, 32)

        # Simulated
        env.MX_EXACT_QUANTIZATION = "False"
        y_simulated = MXTensor.to_mx(x, dtypes.float8_e4m3, 32)

        assert torch.equal(y_hw_exact._data, y_simulated._data)
        assert torch.equal(y_hw_exact._scale_e8m0, y_simulated._scale_e8m0)
        assert torch.equal(
            y_hw_exact.to_dtype(target_dtype), y_simulated.to_dtype(target_dtype)
        )


class TestMXFloat6e3m2:
    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_normal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b1, 0b0111010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor(
            [
                [5, 5, 5, 5, 5, 11],
                [100, 100, 100, 100, 100, 103],
                [250, 250, 250, 250, 250, 251],
            ]
        ).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([7, 99, 247]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [0b0, 0b01, 0b10, 0b0, 0b10, 0], dtype=torch.uint8
        ).view(1, -1)
        gt_exponent = torch.tensor(
            [[2, 1, 1, 1, 1, 7], [5, 4, 4, 4, 4, 7], [7, 6, 6, 6, 6, 7]],
            dtype=torch.uint8,
        )

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 2 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_saturation(self):
        # Construct the input
        bf16_exponent = torch.tensor([100, 100, 100]).to(torch.int16)
        bf16_mantissa = torch.tensor(
            [0b1111010, 0b1110000, 0b1111110], dtype=torch.int16
        )

        bf16_sign = torch.tensor([1, 0, 1], dtype=torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([96]).to(torch.uint8)
        mx_scale = 2.0 ** (shared_exponent.bfloat16() - 127)
        gt = mx_scale * torch.tensor(
            [-dtypes.float6_e3m2.max, dtypes.float6_e3m2.max, -dtypes.float6_e3m2.max],
            dtype=torch.bfloat16,
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 3)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, gt)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_subnormal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b1, 0b0111010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor([100]).to(torch.int16).repeat(3, 6)
        bf16_exponent[0, -1], bf16_exponent[1, -1], bf16_exponent[2, -1] = 109, 108, 107
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([105, 104, 103]).to(torch.uint8).unsqueeze(1)
        gt_mantissa = torch.tensor(
            [
                [0b1, 0b1, 0b1, 0b1, 0b1, 0],
                [0b10, 0b1, 0b10, 0b1, 0b1, 0],
                [0b0, 0b10, 0b11, 0b10, 0b11, 0],
            ]
        ).to(torch.uint8)

        gt_exponent = torch.tensor(
            [[0, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 7], [1, 0, 0, 0, 0, 7]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 2 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_underflow(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b1, 0b0111010, 0], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16)
        bf16_exponent = torch.tensor([100, 100, 100, 100, 100, 110]).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 6)
        y = y_mx.to_dtype(torch.bfloat16)
        # Construct the ground truth
        y_gt = torch.tensor(
            [float("-0"), 0, 0, float("-0"), 0, -(2**-17)], dtype=torch.bfloat16
        )
        assert torch.equal(y, y_gt)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_zeros_to_zeros(self):
        # Construct the input
        bfloat16_inps = torch.zeros(3, 5, dtype=torch.bfloat16)
        bfloat16_inps[0, -1], bfloat16_inps[1, -1], bfloat16_inps[2, -1] = (
            2**-17,
            2**5,
            2**-125,
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 5)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, bfloat16_inps)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_subnormals_to_normal_and_subnormal(self):
        # Construct the input
        bf16_mantissa = (
            torch.tensor(
                [0b1111111, 0b0001010, 0b1000001, 0b0110011, 0b0101010, 0],
                dtype=torch.int16,
            )
            .view(1, -1)
            .expand(3, -1)
        )
        bf16_sign = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.zeros(3, 6).to(torch.int16)
        bf16_exponent[0, -1], bf16_exponent[1, -1], bf16_exponent[2, -1] = 5, 7, 8
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        shared_exponent = torch.tensor([1, 3, 4]).to(torch.uint8)

        # Construct the ground truth
        gt_mantissa = torch.tensor(
            [
                [0b0, 0b1, 0b0, 0b10, 0b01, 0],
                [0b0, 0b0, 0b10, 0b10, 0b1, 0],
                [0b10, 0b0, 0b1, 0b1, 0b1, 0],
            ],
            dtype=torch.uint8,
        )

        gt_exponent = torch.tensor(
            [[3, 0, 2, 1, 1, 7], [1, 0, 0, 0, 0, 7], [0, 0, 0, 0, 0, 7]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 2 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e3m2, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
    def test_random_bfloat16_combinations_hw_exact_and_simulated(
        self, all_bfloat16_values: torch.Tensor, target_dtype: torch.dtype
    ):
        # Generate a random permutation of row indices
        row_indices = torch.randperm(all_bfloat16_values.shape[0])[:-30]
        x = all_bfloat16_values[row_indices]

        # Hardware exact
        env.MX_EXACT_QUANTIZATION = "True"
        y_hw_exact = MXTensor.to_mx(x, dtypes.float6_e3m2, 32)

        # Simulated
        env.MX_EXACT_QUANTIZATION = "False"
        y_simulated = MXTensor.to_mx(x, dtypes.float6_e3m2, 32)

        assert torch.equal(y_hw_exact._data, y_simulated._data)
        assert torch.equal(y_hw_exact._scale_e8m0, y_simulated._scale_e8m0)
        assert torch.equal(
            y_hw_exact.to_dtype(target_dtype), y_simulated.to_dtype(target_dtype)
        )


class TestMXFloat6e2m3:
    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_normal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b1, 0b0111010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor(
            [
                [5, 5, 5, 5, 5, 7],
                [100, 100, 100, 100, 100, 101],
                [250, 250, 250, 250, 250, 250],
            ]
        ).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        shared_exponent = torch.tensor([5, 99, 248]).to(torch.uint8)

        # Construct the ground truth
        gt_mantissa = torch.tensor(
            [0b0, 0b010, 0b100, 0b0, 0b100, 0], dtype=torch.uint8
        ).view(1, -1)
        gt_exponent = torch.tensor(
            [[2, 1, 1, 1, 1, 3], [3, 2, 2, 2, 2, 3], [3, 3, 3, 3, 3, 3]],
            dtype=torch.uint8,
        )

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)
        gt_uint8[2, 0] = 0b111111  # Saturated case

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_saturation(self):
        # Construct the input
        bf16_exponent = torch.tensor([100, 100, 100]).to(torch.int16)
        bf16_mantissa = torch.tensor(
            [0b1111010, 0b1110000, 0b1111110], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 1], dtype=torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([98]).to(torch.uint8)
        mx_scale = 2.0 ** (shared_exponent.bfloat16() - 127)
        gt = mx_scale * torch.tensor(
            [-dtypes.float6_e2m3.max, dtypes.float6_e2m3.max, -dtypes.float6_e2m3.max],
            dtype=torch.bfloat16,
        )
        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 3)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, gt)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_subnormal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0001010, 0b1000001, 0b1, 0b0101010, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor([100]).to(torch.int16).repeat(3, 6)
        bf16_exponent[0, -1], bf16_exponent[1, -1], bf16_exponent[2, -1] = 105, 104, 103
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([103, 102, 101]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [
                [0b10, 0b1, 0b10, 0b1, 0b1, 0],
                [0b100, 0b10, 0b11, 0b10, 0b11, 0],
                [0b0, 0b100, 0b110, 0b100, 0b101, 0],
            ]
        ).to(torch.uint8)

        gt_exponent = torch.tensor(
            [[0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 3]]
        ).to(torch.uint8)

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)
        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_underflow(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b1, 0b0111010, 0], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.int16)
        bf16_exponent = torch.tensor([100, 100, 100, 100, 100, 107]).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )
        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 6)
        y = y_mx.to_dtype(torch.bfloat16)
        # Construct the ground truth
        y_gt = torch.tensor(
            [float("-0"), 0, 0, float("-0"), 0, -(2**-20)], dtype=torch.bfloat16
        )
        assert torch.equal(y, y_gt)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_zeros_to_zeros(self):
        # Construct the input
        bfloat16_inps = torch.zeros(3, 6, dtype=torch.bfloat16)
        bfloat16_inps[0, -1], bfloat16_inps[1, -1], bfloat16_inps[2, -1] = (
            2**-9,
            2**5,
            2**-125,
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 6)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, bfloat16_inps)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_subnormals_to_normal_and_subnormal(self):
        # Construct the input
        bf16_mantissa = (
            torch.tensor(
                [0b1111111, 0b0001010, 0b1000001, 0b0110011, 0b0101010, 0],
                dtype=torch.int16,
            )
            .view(1, -1)
            .expand(2, -1)
        )
        bf16_sign = torch.tensor([0, 1, 0, 0, 1, 1], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.zeros(2, 6).to(torch.int16)
        bf16_exponent[0, -1], bf16_exponent[1, -1] = 2, 3
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([0, 1]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [
                [0b0, 0b1, 0b0, 0b110, 0b101, 0],
                [0b0, 0b1, 0b100, 0b11, 0b11, 0],
            ],
            dtype=torch.uint8,
        )

        gt_exponent = torch.tensor([[2, 0, 1, 0, 0, 3], [1, 0, 0, 0, 0, 3]]).to(
            torch.uint8
        )

        gt_uint8 = (bf16_sign << 5 | gt_exponent << 3 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float6_e2m3, 6)

        assert torch.equal(y_mx._data, gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
    def test_random_bfloat16_combinations_hw_exact_and_simulated(
        self, all_bfloat16_values: torch.Tensor, target_dtype: torch.dtype
    ):
        # Generate a random permutation of row indices
        row_indices = torch.randperm(all_bfloat16_values.shape[0])[:-30]
        x = all_bfloat16_values[row_indices]

        # Hardware exact
        env.MX_EXACT_QUANTIZATION = "True"
        y_hw_exact = MXTensor.to_mx(x, dtypes.float6_e2m3, 32)

        # Simulated
        env.MX_EXACT_QUANTIZATION = "False"
        y_simulated = MXTensor.to_mx(x, dtypes.float6_e2m3, 32)

        assert torch.equal(y_hw_exact._data, y_simulated._data)
        assert torch.equal(y_hw_exact._scale_e8m0, y_simulated._scale_e8m0)
        assert torch.equal(
            y_hw_exact.to_dtype(target_dtype), y_simulated.to_dtype(target_dtype)
        )


class TestMXFloat4e2m1:
    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_normal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b0111010], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 1, 0], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor(
            [[5, 5, 5, 7], [100, 100, 100, 101], [250, 250, 250, 250]]
        ).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        shared_exponent = torch.tensor([5, 99, 248]).to(torch.uint8)

        # Construct the ground truth
        gt_mantissa = torch.tensor([0b0, 0b0, 0b1, 0b1], dtype=torch.uint8).view(1, -1)
        gt_exponent = torch.tensor(
            [[2, 1, 1, 3], [3, 2, 2, 3], [3, 3, 3, 3]], dtype=torch.uint8
        )

        gt_uint8 = (bf16_sign << 3 | gt_exponent << 1 | gt_mantissa).to(torch.uint8)
        gt_uint8[2, 0] = 0b1111  # Saturated case

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 4)

        assert torch.equal(unpack_uint4(y_mx._data), gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_saturation(self):
        # Construct the input
        bf16_exponent = torch.tensor([100, 100, 100, 100]).to(torch.int16)
        bf16_mantissa = torch.tensor(
            [0b1111010, 0b1110000, 0b1111110, 0b1101110], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 1, 0], dtype=torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        shared_exponent = torch.tensor([98]).to(torch.uint8)

        # Construct the ground truth
        mx_scale = 2.0 ** (shared_exponent.bfloat16() - 127)
        gt = mx_scale * torch.tensor(
            [
                -dtypes.float4_e2m1.max,
                dtypes.float4_e2m1.max,
                -dtypes.float4_e2m1.max,
                dtypes.float4_e2m1.max,
            ],
            dtype=torch.bfloat16,
        )
        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 4)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, gt)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_to_subnormal(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0001010, 0b1000001, 0], dtype=torch.int16
        ).view(1, -1)
        bf16_sign = torch.tensor([1, 0, 1, 0], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.tensor([100]).to(torch.int16).repeat(2, 4)
        bf16_exponent[0, -1], bf16_exponent[1, -1] = 104, 103
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        shared_exponent = torch.tensor([102, 101]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [
                [0b1, 0b1, 0b1, 0b0],
                [0b0, 0b1, 0b0, 0b0],
            ]
        ).to(torch.uint8)

        gt_exponent = torch.tensor([[0, 0, 0, 3], [1, 0, 1, 3]]).to(torch.uint8)

        gt_uint8 = (bf16_sign << 3 | gt_exponent << 1 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 4)

        assert torch.equal(unpack_uint4(y_mx._data), gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_normal_underflow(self):
        # Construct the input
        bf16_mantissa = torch.tensor(
            [0b1111111, 0b0011010, 0b1000001, 0b0111010, 0, 0], dtype=torch.int16
        )
        bf16_sign = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int16)
        bf16_exponent = torch.tensor([100, 100, 100, 100, 100, 105]).to(torch.int16)
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 6)
        y = y_mx.to_dtype(torch.bfloat16)
        # Construct the ground truth
        y_gt = torch.tensor(
            [float("-0"), 0, 0, float("-0"), 0, 2**-22], dtype=torch.bfloat16
        )
        assert torch.equal(y, y_gt)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_zeros_to_zeros(self):
        # Construct the input
        bfloat16_inps = torch.zeros(3, 6, dtype=torch.bfloat16)
        bfloat16_inps[0, -1], bfloat16_inps[1, -1], bfloat16_inps[2, -1] = (
            2**-9,
            2**5,
            2**-125,
        )

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 6)
        y = y_mx.to_dtype(torch.bfloat16)

        assert torch.equal(y, bfloat16_inps)

    @pytest.mark.usefixtures("set_quantization_env")
    def test_bf16_subnormals_to_normal_and_subnormal(self):
        # Construct the input
        bf16_mantissa = (
            torch.tensor(
                [0b1111111, 0b0011010, 0b1000001, 0b0110011, 0b0101010, 0],
                dtype=torch.int16,
            )
            .view(1, -1)
            .expand(2, -1)
        )
        bf16_sign = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int16).view(1, -1)
        bf16_exponent = torch.zeros(2, 6).to(torch.int16)
        bf16_exponent[0, -1], bf16_exponent[1, -1] = 2, 3
        bfloat16_inps = (bf16_sign << 15 | bf16_exponent << 7 | bf16_mantissa).view(
            torch.bfloat16
        )

        # Construct the ground truth
        # # normalized mantissa: [0b1111110, 0b1010000, 0b0000010, 0b1001100, 0b0101000, 0b0001000]
        shared_exponent = torch.tensor([0, 1]).to(torch.uint8)
        gt_mantissa = torch.tensor(
            [
                [0b0, 0b1, 0b0, 0b0, 0b1, 0b0],
                [0b0, 0b0, 0b1, 0b1, 0b1, 0b0],
            ],
            dtype=torch.uint8,
        )

        gt_exponent = torch.tensor([[2, 0, 1, 1, 0, 3], [1, 0, 0, 0, 0, 3]]).to(
            torch.uint8
        )

        gt_uint8 = (bf16_sign << 3 | gt_exponent << 1 | gt_mantissa).to(torch.uint8)

        # Call mx quantization
        y_mx = MXTensor.to_mx(bfloat16_inps, dtypes.float4_e2m1, 6)

        assert torch.equal(unpack_uint4(y_mx._data), gt_uint8)
        assert torch.equal(y_mx._scale_e8m0, shared_exponent.unsqueeze(1))

    @pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
    def test_random_bfloat16_combinations_hw_exact_and_simulated(
        self, all_bfloat16_values: torch.Tensor, target_dtype: torch.dtype
    ):
        # Generate a random permutation of row indices
        row_indices = torch.randperm(all_bfloat16_values.shape[0])[:-30]
        x = all_bfloat16_values[row_indices]

        # Hardware exact
        env.MX_EXACT_QUANTIZATION = "True"
        y_hw_exact = MXTensor.to_mx(x, dtypes.float4_e2m1, 32)

        # Simulated
        env.MX_EXACT_QUANTIZATION = "False"
        y_simulated = MXTensor.to_mx(x, dtypes.float4_e2m1, 32)

        assert torch.equal(y_hw_exact._data, y_simulated._data)
        assert torch.equal(y_hw_exact._scale_e8m0, y_simulated._scale_e8m0)
        assert torch.equal(
            y_hw_exact.to_dtype(target_dtype), y_simulated.to_dtype(target_dtype)
        )


class TestRoundToEven:
    """
    Test round to even function
    """

    def test_basic_rounding(self):
        # Test case where rounding is needed
        mantissa = torch.tensor([0b1010011, 0b1101101], dtype=torch.int32)  # [83, 109]
        mantissa_shift = torch.tensor([2, 3], dtype=torch.int32)
        expected = torch.tensor([21, 14], dtype=torch.int32)  # Expected rounding
        output_tensorized = round_to_even(mantissa, mantissa_shift)
        torch.testing.assert_close(output_tensorized, expected)

    def test_no_rounding_needed(self):
        # Test case where no rounding is needed
        mantissa = torch.tensor([0b1010000, 0b1100000], dtype=torch.int32)  # [80, 96]
        mantissa_shift = torch.tensor([2, 3], dtype=torch.int32)
        expected = torch.tensor([20, 12], dtype=torch.int32)  # No rounding needed
        output_tensorized = round_to_even(mantissa, mantissa_shift)
        torch.testing.assert_close(output_tensorized, expected)

    def test_round_half_to_even(self):
        # Test the "round half to even" behavior
        mantissa = torch.tensor([0b1010110, 0b1101100], dtype=torch.int32)  # [86, 104]
        mantissa_shift = torch.tensor([2, 3], dtype=torch.int32)
        expected = torch.tensor([22, 14], dtype=torch.int32)  # Rounded to even
        output_tensorized = round_to_even(mantissa, mantissa_shift)
        torch.testing.assert_close(output_tensorized, expected)

    def test_all_zero_mantissa(self):
        # Test case with all zero mantissa
        mantissa = torch.tensor([0, 0], dtype=torch.int32)
        mantissa_shift = torch.tensor([2, 3], dtype=torch.int32)
        expected = torch.tensor([0, 0], dtype=torch.int32)  # No rounding needed
        output_tensorized = round_to_even(mantissa, mantissa_shift)
        torch.testing.assert_close(output_tensorized, expected)

    def test_shift_zero(self):
        # Test case where shift is zero (no shift)
        mantissa = torch.tensor([0b1010011, 0b1101101], dtype=torch.int32)  # [83, 109]
        mantissa_shift = torch.tensor([0, 0], dtype=torch.int32)
        output_tensorized = round_to_even(mantissa, mantissa_shift)
        torch.testing.assert_close(output_tensorized, mantissa)
