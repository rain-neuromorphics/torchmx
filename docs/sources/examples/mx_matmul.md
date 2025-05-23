# MXTensor Matmul example

This script tests matrix multiplication operations using MXTensor from the `torchmx` library. It generates random tensors, converts them into MXTensor format, and performs a matrix multiplication on the MXTensor using `torch.matmul`.

```python
import torch

from torchmx import dtypes
from torchmx.mx_tensor import MXTensor
from torchmx.utils import get_logger, get_uniform_random_number

logger = get_logger("check_mxtensor_ops")


def main():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
    else:
        DEVICE = torch.device("cpu")
    DTYPE = torch.bfloat16
    logger.info(f"using device: {DEVICE}")
    a = get_uniform_random_number(0, 10, (128, 256), DTYPE).to(DEVICE)
    b = get_uniform_random_number(0, 10, (256, 512), DTYPE).to(DEVICE)
    mx_a = MXTensor.to_mx(a, elem_dtype=dtypes.float8_e4m3, block_size=32)
    mx_b = MXTensor.to_mx(b, elem_dtype=dtypes.float8_e4m3, block_size=32)

    c = torch.matmul(mx_a, mx_b)
    logger.info(f"matmul result shape: {c.shape}")
    assert isinstance(c, torch.Tensor) and not isinstance(c, MXTensor)


if __name__ == "__main__":
    main()

```
