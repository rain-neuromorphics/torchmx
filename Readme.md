# TorchMX: PyTorch Quantization Framework For OCP MX Datatypes

This package a simulation tool implementing the MX quantization format specified in the
[OCP Micro Scaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). The pacakage includes:

* Tensor subclasses for representing MX quantized data `MXTensor`.
* Quantization and dequantization functions for converting between high-precision and MX quantized tensors.
* Support for various MX data types, including FP8, FP6, FP4, and INT8.
* Custom `ATen` operations for `MXTensor`

## Installation

```bash
pip install torchmx
```

## Usage

Here's a basic example of how to quantize a PyTorch tensor to MX format:

```python
import torch
from torchmx import MXTensor, dtypes

# Create a high-precision tensor
x_hp = torch.randn(128, 128, dtype=torch.bfloat16)

# Quantize the tensor to MX format
x_mx = MXTensor.to_mx(x_hp, elem_dtype=dtypes.float8_e4m3, block_size=32)

# Dequantize the tensor back to high-precision
x_hp_reconstructed = x_mx.to_dtype(torch.bfloat16)

# Matmul 2 MXTensors
y_hp = torch.randn(128, 128, dtype=torch.bfloat16)
y_mx = MXTensor.to_mx(y_mx, elem_dtype=dtypes.float6_e3m2, block_size=32)

# Notice the magic here. You can pass MXTensors into torch.matmul.
# This even works for 4D Attention Matmuls torch.matmul(Q, K.t).
# The output is a bf16 torch.Tensor
out_bf16 = torch.matmul(x_mx, y_mx)
```

## Quantizing Layers and Modules

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

## Examples

For more detailed examples refer the [examples](https://github.com/rain-neuromorphics/torchmx/tree/main/examples) directory

### Testing

```bash
pytest
```
