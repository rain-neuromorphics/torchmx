# Quantizing Linear Layers in a model

TorchMX also provides tools for quantizing individual layers and modules. Here's an example of how to quantize all the linear layers in the model. The following example demonstrates how to quantize a model with torch.nn.Linear layers to MX format using the MXInferenceLinear class. In this case the weights are quantized `MX-fp6_e3m2` and the
inputs to `MX-fp8_e4m3`

```python
from torch import nn
from torchmx import MXTensor, dtypes
from torchmx.config import QLinearConfig, MXConfig
from torchmx.quant_api import quantize_linear_

# Create a high-precision model
model = nn.Sequential(
          nn.Linear(128, 256),
          nn.ReLU(),
          nn.Linear(256, 64),
          nn.ReLU()
        ).to(torch.bfloat16)

# Define the quantization configuration
qconfig = QLinearConfig(
    weights_config=MXConfig(elem_dtype_name="float6_e3m2", block_size=32),
    activations_config=MXConfig(elem_dtype_name="float8_e4m3", block_size=32),
)

# Quantize the model to MXFormat. Note this quantizes the model in place
quantize_linear_(model=model, qconfig=qconfig)


# Perform inference using the quantized model
x_hp = torch.randn(16, 128, dtype=torch.bfloat16)
y_mx = model(x_hp)
```
