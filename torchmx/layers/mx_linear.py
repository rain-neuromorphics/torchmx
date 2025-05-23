import torch
import torch.nn.functional as F

from ..config import QLinearConfig
from ..mx_tensor import MXTensor


class MXInferenceLinear(torch.nn.Linear):
    """
    Inference version of MXLinear, with the weight pre-quantized to MX.
    """

    def extra_repr(self):
        return ", ".join(
            (
                super().extra_repr(),
                f"qconfig={self.qconfig}",
            )
        )

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        qconfig: QLinearConfig,
    ) -> "MXInferenceLinear":
        """Converts a torch.nn.Linear module to MXInferenceLinear.

        Args:
            mod (torch.nn.Linear): The module to convert.
            qconfig (QLinearConfig): The quantization configuration.

        Returns:
            MXInferenceLinear: The converted module.
        """
        with torch.device("meta"):
            super_kwargs = {
                "in_features": mod.in_features,
                "out_features": mod.out_features,
                "bias": False,
            }
            new_mod = cls(**super_kwargs)
        new_mod.qconfig = qconfig
        if not mod.weight.data.device == torch.device("meta"):
            # Do NOT quantize on `meta` device as it causes issues with `accelerate`
            # This line fails: https://github.com/huggingface/accelerate/blob/b431d1f02733dfcb1668c7f3cffe1e2f25fb94e7/src/accelerate/utils/modeling.py#L366
            # They manually patch for `torchao`: https://github.com/huggingface/accelerate/blob/b431d1f02733dfcb1668c7f3cffe1e2f25fb94e7/src/accelerate/utils/modeling.py#L353
            # The simpler solution for now is to not quantize here but to quantize everytime during the forward pass when the weights are on the meta device.
            new_mod.weight = torch.nn.Parameter(
                MXTensor.to_mx(
                    mod.weight.data,
                    qconfig.weights_config.elem_dtype,
                    qconfig.weights_config.block_size,
                ),
                requires_grad=False,
            )
        new_mod.bias = mod.bias
        return new_mod

    @torch.no_grad()
    def forward(self, x):
        x_mx = MXTensor.to_mx(
            x,
            self.qconfig.activations_config.elem_dtype,
            self.qconfig.activations_config.block_size,
        )
        if not isinstance(self.weight.data, MXTensor):
            # This case happens when we are performing big model inference using
            # `accelerate` and the weights are on the disk, `accelerate` puts the
            # weights on the `meta` device. For more information, refer to the comment
            # in the from_float() above.

            # Do not modify the self.weight and directly send the quantized weights to
            # F.linear() so that casting back to `meta` device works correctly inside
            # `accelerate`.

            # For some reason the weights in the disk when loaded are in `torch.float32`
            # and do `NOT` respect the dtype passed when creating the HF model. So we
            # have to cast them to the correct dtype. We only support MX quant for
            # torch.bfloat16.

            # Cast bias to torch.bfloat16 for safety
            return F.linear(
                x_mx,
                MXTensor.to_mx(
                    self.weight.data.to(torch.bfloat16),
                    self.qconfig.weights_config.elem_dtype,
                    self.qconfig.weights_config.block_size,
                ),
                self.bias.to(torch.bfloat16) if self.bias else self.bias,
            )
        else:
            y = F.linear(x_mx, self.weight, self.bias)
        return y
