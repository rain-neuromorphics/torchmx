# LLaMA Quantized Inference Results

This section presents empirical results on the application of `TorchMX` to the LLaMA 3.1 series of models, specifically the 8B and 70B variants. Our objective is to evaluate the efficacy of quantization using the Microscaling Floating Point (MXFP) format, which allows low-bit inference across all major tensor operations. We demonstrate that TorchMX enables near-lossless inference—achieving sub-2% accuracy degradation—without requiring post-training calibration.

---

## Quantization Setup

We apply MXFP quantization with a block size of 32 to the following components:

* All weights and activations in projection and MLP layers
* Query, Key, and Value (QKV) vectors
* Attention weight matrices (used in matmul with Value)

Matrix multiplications and softmax layers are computed in dequantized `bfloat16`.

---

## Evaluation Setup

* **Models Evaluated**: LLaMA 3.1-8B, LLaMA 3.1-70B
* **Datasets**: PIQ, ARC Easy, ARC Challenge, HellaSwag, Winogrande
* **Baseline Precision**: `bfloat16`
* **Inference Hardware**: NVIDIA A100 80GB

---

## Accuracy Comparison

| Model                | ProjW                                 | ProjA                                 | MlpW                                  | MlpA                                  | Query                                 | Key                                   | Value                                 | Atten W                               | Aver. Acc. (%) | Acc. Δ (%) |
| -------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | ------------------------------------- | -------------- | ---------- |
| LLaMA 3.1-8B (bf16)  | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | 73.60          | —          |
|                      | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | -                                     | -                                     | -                                     | -                                     | 73.26          | -0.34      |
|                      | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | -                                     | -                                     | -                                     | -                                     | 73.12          | -0.48      |
|                      | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | 71.79          | -1.81      |
|                      | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | 71.76          | -1.84      |
| LLaMA 3.1-70B (bf16) | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | -                                     | 79.93          | —          |
|                      | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | -                                     | -                                     | -                                     | -                                     | 79.35          | -0.58      |
|                      | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | -                                     | -                                     | -                                     | -                                     | 78.94          | -1.00      |
|                      | <span style="color:orange">FP6</span> | <span style="color:blue">FP8</span>   | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | 78.63          | -1.30      |
|                      | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | <span style="color:orange">FP6</span> | 78.63          | -1.47      |

---

## Analysis and Insights

* **Projection activations** appear more sensitive to quantization than MLP activations, especially under FP6.
* A full-stack FP6 configuration achieves excellent tradeoffs, showing just \~1.8% degradation while offering substantial compression.
* Using FP8 for activations (especially in projection) recovers up to 0.2–0.5% accuracy in both 8B and 70B variants.
* Value vectors can be further compressed (e.g., MXFP4) with negligible loss (results not shown).

---

## Reproducibility

To replicate these benchmarks, see the example script:

```bash
examples/quantize_llama.md
```
