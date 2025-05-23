import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..config import QAttentionConfig, QLinearConfig
from ..mx_tensor import MXTensor

from .mx_linear import MXInferenceLinear


class MXInferenceLlamaMLP(LlamaMLP):
    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: LlamaMLP,
        qconfig: QLinearConfig,
    ) -> "MXInferenceLlamaMLP":
        """Converts LlamaMLP from transformers to MXInferenceLlamaMLP.

         Args:
            mod (LlamaMLP): The module to convert.
            qconfig (QLinearConfig): The quantization configuration.


        Returns:
             MXInferenceLlamaMLP: The converted module.
        """
        assert isinstance(
            mod, LlamaMLP
        ), f"mod must be an instance of LlamaMLP, but got {type(mod)}"
        with torch.device("meta"):
            super_kwargs = {
                "config": mod.config,
            }
            qlmlp = cls(**super_kwargs)

        qlmlp.qconfig = qconfig
        qlmlp.gate_proj = MXInferenceLinear.from_float(
            mod=mod.gate_proj,
            qconfig=qconfig,
        )
        qlmlp.up_proj = MXInferenceLinear.from_float(
            mod=mod.up_proj,
            qconfig=qconfig,
        )
        qlmlp.down_proj = MXInferenceLinear.from_float(
            mod=mod.down_proj,
            qconfig=qconfig,
        )
        return qlmlp


class MXInferenceLlamaAttention(LlamaAttention):
    """The MX inference version of LlamaAttention from transformers."""

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: LlamaAttention,
        qconfig: QAttentionConfig,
    ) -> "MXInferenceLlamaAttention":
        """Converts LlamaAttention from transformers to MXInferenceLlamaAttention.

        Args:
            mod (LlamaAttention): The module to convert.
            qconfig (QAttentionConfig): The quantization configuration.
        Returns:
            MXInferenceLlamaAttention: The converted module.

        """
        assert isinstance(
            mod, LlamaAttention
        ), f"mod must be an instance of LlamaAttention, but got {type(mod)}"

        with torch.device("meta"):
            super_kwargs = {
                "config": mod.config,
                "layer_idx": mod.layer_idx,
            }
            qlla = cls(**super_kwargs)

        # Transfer key, value and query layer parameters
        qlla.qconfig = qconfig

        qlla.q_proj = MXInferenceLinear.from_float(
            mod=mod.q_proj,
            qconfig=qconfig.projection_config,
        )
        qlla.k_proj = MXInferenceLinear.from_float(
            mod=mod.k_proj,
            qconfig=qconfig.projection_config,
        )
        qlla.v_proj = MXInferenceLinear.from_float(
            mod=mod.v_proj,
            qconfig=qconfig.projection_config,
        )
        qlla.o_proj = MXInferenceLinear.from_float(
            mod=mod.o_proj, qconfig=qconfig.projection_config
        )
        return qlla

    def extra_repr(self):
        return ", ".join(
            (
                super().extra_repr(),
                f"qconfig={self.qconfig}",
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            # https://github.com/huggingface/transformers/blob/7d4b3ddde4da3cba17403072cfdb8b8c76ca1c7c/src/transformers/models/llama/configuration_llama.py#L74
            raise NotImplementedError("Pretraining TP > 1 is not supported yet.")
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        # Nomecluture:
        # (dim0, dim1, -dim2-): -dim2- denotes that MX quantization is applied along this dimension
        # (dim0, *dim1*, dim2): *dim1* denotes accumulation dimension for matmul
        # (dim0, dim1, -*dim2*-): -*dim2*- denotes MX quantization is applied along the accumulation dimension

        # (bs, num_heads, q_len, head_dim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # (bs, num_key_value_heads, q_len, head_dim)
        # num_key_value_heads < num_heads for Grouped Query Attention
        # num_key_value_heads = num_heads for Multi Head attention
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # (bs, num_key_value_heads, q_len, head_dim): Grouped query attention
        # num_key_value_heads < num_heads for Grouped Query Attention
        # num_key_value_heads = num_heads for Multi Head attention
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # Cache is updated and stored in High precision.
            # TODO: KV cache quantization is NOT implemented yet.
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # repeat_kv is used to repeat the key and value states to make matmul work for the grouped query attention
        # It repeats num_key_value_heads such that the num_key_value_heads = num_heads
        # num_key_value_heads * num_key_value_groups = num_heads - For more info look at the __init__() of LlamaAttention in transformers
        # It is a no-op for the multi-head attention
        # (bs, num_heads, q_len, head_dim):
        key_states = repeat_kv(key_states, self.num_key_value_groups)

        # (bs, num_heads, q_len, head_dim)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        query_states_orig_dtype = query_states.dtype
        if self.qconfig.is_qkv_quantization_enabled:
            # (bs, num_heads, q_len, -head_dim-)
            query_states = MXTensor.to_mx(
                data_hp=query_states.contiguous(),
                elem_dtype=self.qconfig.query_config.elem_dtype,
                block_size=self.qconfig.query_config.block_size,
            )
            # (bs, num_key_value_heads, q_len, -head_dim-)
            key_states = MXTensor.to_mx(
                data_hp=key_states.contiguous(),
                elem_dtype=self.qconfig.key_config.elem_dtype,
                block_size=self.qconfig.key_config.block_size,
            )
            # (bs, num_heads, -q_len- , head_dim)
            value_states = MXTensor.to_mx(
                data_hp=value_states.transpose(2, 3).contiguous(),
                elem_dtype=self.qconfig.value_config.elem_dtype,
                block_size=self.qconfig.value_config.block_size,
            ).transpose(2, 3)
        # (bs, num_heads, q_len, -*head_dim*-) *(bs, num_key_value_heads, -*head_dim(-, q_len) = (bs, num_heads, q_len, q_len)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states_orig_dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Quantize attn_weights if kv_cache_config is provided
        # Right now we are quantizing after softmax and dropout
        # TODO: Once we have approximate softmax, quantize before softmax.
        if self.qconfig.is_qkv_quantization_enabled:
            # (bs, num_heads, q_len, -q_len-)
            attn_weights = MXTensor.to_mx(
                data_hp=attn_weights,
                elem_dtype=self.qconfig.attention_weights_config.elem_dtype,
                block_size=self.qconfig.attention_weights_config.block_size,
            )

        # (bs, num_heads, q_len, -*q_len*-) * (bs, num_heads, -*q_len*- , head_dim) = (bs, num_heads, q_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # (bs, q_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # (bs, q_len, num_heads * head_dim)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Pretraining TP > 1 is not supported yet.")
        else:
            # (bs, q_len, -num_heads * head_dim-)
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
