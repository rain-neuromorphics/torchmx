import math
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)

from ..config import QAttentionConfig, QLinearConfig
from ..mx_tensor import MXTensor

from .mx_linear import MXInferenceLinear


class MXInferenceQwen2MLP(Qwen2MLP):
    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: Qwen2MLP,
        qconfig: QLinearConfig,
    ) -> "MXInferenceQwen2MLP":
        """Converts Qwen2MLP from transformers to MXInferenceQwen2MLP.

         Args:
            mod (Qwen2MLP): The module to convert.
            qconfig (QLinearConfig): The quantization configuration.


        Returns:
             MXInferenceQwen2MLP: The converted module.
        """
        assert isinstance(
            mod, Qwen2MLP
        ), f"mod must be an instance of Qwen2MLP, but got {type(mod)}"

        # HF forgot to save the config object. This is a hack
        config = Qwen2Config(
            hidden_size=mod.hidden_size,
            intermediate_size=mod.intermediate_size,
        )
        with torch.device("meta"):
            super_kwargs = {
                "config": config,
            }
            qlmlp = cls(**super_kwargs)

        assert type(mod.act_fn) == type(
            qlmlp.act_fn
        ), f"mod.act_fn: {type(mod.act_fn)}, qlmlp.act_fn: {type(qlmlp.act_fn)}"

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


class MXInferenceQwen2Attention(Qwen2Attention):
    """The MX inference version of Qwen2Attention from transformers."""

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod: Qwen2Attention,
        qconfig: QAttentionConfig,
    ) -> "MXInferenceQwen2Attention":
        """Converts Qwen2Attention from transformers to MXInferenceQwen2Attention.

        Args:
            mod (Qwen2Attention): The module to convert.
            qconfig (QAttentionConfig): The quantization configuration.
        Returns:
            MXInferenceQwen2Attention: The converted module.

        """
        assert isinstance(
            mod, Qwen2Attention
        ), f"mod must be an instance of Qwen2Attention, but got {type(mod)}"

        with torch.device("meta"):
            super_kwargs = {
                "config": mod.config,
                "layer_idx": mod.layer_idx,
            }
            qqwen = cls(**super_kwargs)

        # Transfer key, value and query layer parameters
        qqwen.qconfig = qconfig

        qqwen.q_proj = MXInferenceLinear.from_float(
            mod=mod.q_proj,
            qconfig=qconfig.projection_config,
        )
        qqwen.k_proj = MXInferenceLinear.from_float(
            mod=mod.k_proj,
            qconfig=qconfig.projection_config,
        )
        qqwen.v_proj = MXInferenceLinear.from_float(
            mod=mod.v_proj,
            qconfig=qconfig.projection_config,
        )
        qqwen.o_proj = MXInferenceLinear.from_float(
            mod=mod.o_proj, qconfig=qconfig.projection_config
        )

        qqwen.rotary_emb = Qwen2RotaryEmbedding(
            qqwen.head_dim,
            max_position_embeddings=qqwen.max_position_embeddings,
            base=qqwen.rope_theta,
            device=mod.q_proj.weight.data.device,
        )
        return qqwen

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
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

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

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

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
