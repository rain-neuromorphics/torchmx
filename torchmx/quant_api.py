"""Quantization API for LLM models."""

import functools
import types
from pprint import pformat

import torch
from torchao.dtypes import AffineQuantizedTensor
from torchao.quantization.linear_activation_quantized_tensor import (
    LinearActivationQuantizedTensor,
    to_linear_activation_quantized,
)
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP

from . import dtypes
from .config import QAttentionConfig, QLinearConfig
from .layers.mx_linear import MXInferenceLinear
from .layers.mx_llama_attention import (
    MXInferenceLlamaAttention,
    MXInferenceLlamaMLP,
)
from .layers.mx_qwen2_attention import (
    MXInferenceQwen2Attention,
    MXInferenceQwen2MLP,
)
from .mx_tensor import MXTensor
from .utils import get_logger

aten = torch.ops.aten

logger = get_logger(__name__)


def _quantization_type(weight: torch.Tensor):
    if isinstance(weight, AffineQuantizedTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if isinstance(weight, LinearActivationQuantizedTensor):
        return f"{weight.__class__.__name__}(activation={weight.input_quant_func}, weight={_quantization_type(weight.original_weight_tensor)})"

    if isinstance(weight, MXTensor):
        return f"{weight.__class__.__name__}({weight._quantization_type()})"

    if type(weight) is torch.Tensor:
        return "not quantized"

    return "not recognized"


def _linear_extra_repr(self):
    return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, weight={_quantization_type(self.weight)}"


def _get_linear_subclass_inserter(constructor, *, allow_requires_grad=False, **kwargs):
    """Helper function to apply the constructor that quantizes the weight Tensor (with additional kwargs)
    to the weight of linear module
    """

    def insert_subclass(lin):
        requires_grad = allow_requires_grad and lin.weight.requires_grad
        lin.weight = torch.nn.Parameter(
            constructor(lin.weight, **kwargs), requires_grad=requires_grad
        )
        lin.extra_repr = types.MethodType(_linear_extra_repr, lin)
        return lin

    return insert_subclass


def _apply_mx_dynamic_activation_mx_weights(
    weight: torch.Tensor,
    weight_elem_dtype: dtypes.DType = dtypes.float6_e3m2,
    weight_block_size: int = 32,
    activation_elem_dtype: dtypes.DType = dtypes.float8_e4m3,
    activation_block_size: int = 32,
):
    """Present for serialization purposes."""
    if weight.numel() % weight_block_size != 0:
        raise ValueError(
            f"Weight tensor size {weight.numel()} is not divisible by block size {weight_block_size}"
        )
    weight = MXTensor.to_mx(
        data_hp=weight, elem_dtype=weight_elem_dtype, block_size=weight_block_size
    )
    activation_quant_func = functools.partial(
        MXTensor.to_mx,
        elem_dtype=activation_elem_dtype,
        block_size=activation_block_size,
    )
    weight = to_linear_activation_quantized(weight, activation_quant_func)
    return weight


def mx_dynamic_activation_mx_weights(
    weight_elem_dtype: dtypes.DType = dtypes.float6_e3m2,
    weight_block_size: int = 32,
    activation_elem_dtype: dtypes.DType = dtypes.float8_e4m3,
    activation_block_size: int = 32,
):
    """Quantize the model with MXFP Dynamic quantization for activations and MXFP
    quantization for weights. This directly replaces the nn.Linear module's weight param
    This is a helper function to be used with `torchao.quantization.quantize_`.
    You can use this if you want to quantize all Linear layers in the model with MXFP
    Dynamic quantization and do not want to make a distinction between Attention and MLP
    See below for an example of how to use this function.

    Args:
        weight_elem_dtype (dtypes.DType, optional): Weight element dtype. Defaults to dtypes.float6_e3m2.
        weight_block_size (int, optional): Weight block size. Defaults to 32.
        activation_elem_dtype (dtypes.DType, optional): Activation element dtype. Defaults to dtypes.float8_e4m3.
        activation_block_size (int, optional): Activation block size. Defaults to 32.

    Usage:
    ```python
    import torchao

    model = LLM()
    torchao.quantization.quantize_(
        model,
        mx_dynamic_activation_mx_weights(
            weight_elem_dtype=dtypes.float6_e3m2,
            weight_block_size=weight_block_size,
            activation_elem_dtype=dtypes.float8_e4m3,
            activation_block_size=activation_block_size,
        ),
    )
    print(f"Quantized model: {model}")
    ```
    """
    logger.info(
        "Quantizing model with MXFP Dynamic quantization for activations and MXFP quantization for weights"
    )
    logger.info(
        f"Weight element dtype: {weight_elem_dtype}, Weight block size: {weight_block_size}"
    )
    logger.info(
        f"Activation element dtype: {activation_elem_dtype}, Activation block size: {activation_block_size}"
    )
    return _get_linear_subclass_inserter(
        _apply_mx_dynamic_activation_mx_weights,
        weight_elem_dtype=weight_elem_dtype,
        weight_block_size=weight_block_size,
        activation_elem_dtype=activation_elem_dtype,
        activation_block_size=activation_block_size,
    )


ATTENTION_LAYERS = {
    LlamaAttention: MXInferenceLlamaAttention,
    Qwen2Attention: MXInferenceQwen2Attention,
}

MLP_LAYERS = {
    LlamaMLP: MXInferenceLlamaMLP,
    Qwen2MLP: MXInferenceQwen2MLP,
}


def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    pbar: tqdm,
    cur_fqn="",
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        pbar.update(1)
        if cur_fqn == "":
            new_fqn = name
        else:
            new_fqn = f"{cur_fqn}.{name}"
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            _replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, pbar, new_fqn
            )


def quantize_linear_(model: torch.nn.Module, qconfig: QLinearConfig):
    """Quantize an LLM by swapping the Linear layers in place

    This method only replaces/quantizes the linear layers. Use this as an approximation
    as we do not quantize QKV and other stuff. Use this only when a specific attention layer is not implemented.

    Args:
        model (torch.nn.Module): The model to quantize.
        qconfig (QLLMConfig): The quantization configuration.
    """
    logger.info(
        "Quantizing the model by swapping nn.Linear with TorchMX's MXInferenceLinear"
    )
    logger.warning(
        "This method only replaces/quantizes the linear layers. Use this as an approximation as we do not quantize QKV and other stuff. Use this only when a specific attention layer is not implemented."
    )
    logger.info(f"Quantizing Linear layers with config:\n{pformat(qconfig)}\n")
    _replace_with_custom_fn_if_matches_filter(
        model=model,
        replacement_fn=lambda mod: MXInferenceLinear.from_float(
            mod=mod,
            qconfig=qconfig,
        ),
        filter_fn=lambda mod, fqn: type(mod) == torch.nn.Linear,
        pbar=tqdm(
            desc="Quantizing linear layers in model...",
        ),
    )


def quantize_llm_(
    model: torch.nn.Module,
    qattention_config: QAttentionConfig,
    qmlp_config: QLinearConfig,
):
    """Quantize the LLM by swapping the Attention Layer and MLP Layer in place.
    The implemented Layers is expected to handle all possible quantization layers.
    Refer to `torchmx/layers/mx_llama_attention.py` for more details.

    Args:
        model (torch.nn.Module): The model to quantize.
        qattention_config (QAttentionConfig): The quantization configuration for the attention layers.
        qmlp_config (QLinearConfig): The quantization configuration for the MLP layers.
    """
    logger.info("Quantizing the model by swapping the Attention and MLP layers")
    logger.info(f"Attention Layer Quantization config:\n{pformat(qattention_config)}\n")
    logger.info(f"MLP Layer Quantization config:\n{pformat(qmlp_config)}\n")
    LAYERS_TO_REPLACE = {}
    LAYERS_TO_REPLACE.update(ATTENTION_LAYERS)
    LAYERS_TO_REPLACE.update(MLP_LAYERS)

    def _filter_fn(mod, fqn):
        if type(mod) in LAYERS_TO_REPLACE:
            return True
        else:
            return False

    def _replacement_fn(mod):
        replace_mod = LAYERS_TO_REPLACE[type(mod)]
        if replace_mod in ATTENTION_LAYERS.values():
            return replace_mod.from_float(
                mod,
                qconfig=qattention_config,
            )
        elif replace_mod in MLP_LAYERS.values():
            return replace_mod.from_float(
                mod,
                qconfig=qmlp_config,
            )
        else:
            raise ValueError(f"Unsupported layer type: {replace_mod}")

    _replace_with_custom_fn_if_matches_filter(
        model=model,
        replacement_fn=_replacement_fn,
        filter_fn=_filter_fn,
        pbar=tqdm(
            desc="Quantizing Attention and MLP layers in model...",
        ),
    )
    quantize_linear_(
        model=model,
        qconfig=qmlp_config,
    )
