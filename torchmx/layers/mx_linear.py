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
        y = F.linear(x_mx, self.weight, self.bias)
        return y
