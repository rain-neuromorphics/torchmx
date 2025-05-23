# **DType Constants for PyTorch MX Formats**

## Overview

This module defines the `DType` class and various numerical data types used in PyTorch's MX formats. It includes information about their properties, such as exponent bias, mantissa bits, and maximum representable values.

---

## **DType Class Definition**

```python
@dataclass(frozen=True, repr=False)
class DType:
    name: str
    max: float  # Maximum representable value
    max_pow2: int  # Largest power of 2 representable: floor(log2(max))
    exponent_bias: int  # Exponent bias
    exponent_bits: int  # Number of exponent bits
    mantissa_bits: int  # Number of mantissa bits
    torch_dtype: Optional[torch.dtype] = None  # Corresponding torch.dtype if available

    def __repr__(self):
        return self.name
```

---

## **Supported DTypes for MX Format**

All the `data types` below are objects of the above `DType` class. You can use any of the following as input to `elem_dtype`

### **Float Types**

| Name            | Max Value                                                     | Max Pow2 | Exponent Bias | Exponent Bits | Mantissa Bits | PyTorch DType         |
|-----------------|---------------------------------------------------------------|----------|---------------|---------------|---------------|-----------------------|
| `float8_e4m3`   | 448.0                                                         | 8        | 7             | 4             | 3             | `torch.float8_e4m3fn` |
| `float6_e3m2`   | 28.0                                                          | 4        | 3             | 3             | 2             | None                  |
| `float6_e2m3`   | 7.5                                                           | 2        | 1             | 2             | 3             | None                  |
| `float4_e2m1`   | 6.0                                                           | 2        | 1             | 2             | 1             | None                  |

### **Integer Types**

| Name  | Max Value | Max Pow2 | Exponent Bias | Exponent Bits | Mantissa Bits | PyTorch DType |
|-------|----------|----------|---------------|--------------|--------------|----------------|
| `int8` | 127.0  | 6        | 0             | 0            | 7            | `torch.int8` |

---

## Other convenient variables

### Supported Element Types

```python
SUPPORTED_ELEM_DTYPES = (
    float8_e4m3,
    float6_e3m2,
    float6_e2m3,
    float4_e2m1,
    int8,
)
```


---

## Mappings for Easy Lookup

```python
STR_TO_SUPPORTED_ELEM_DTYPE = {d.name: d for d in SUPPORTED_ELEM_DTYPES}
```

---
