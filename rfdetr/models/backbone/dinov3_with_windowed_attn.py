# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from HuggingFace Dinov3 (https://github.com/huggingface/transformers)
# Copyright 2024 Meta Inc. and the HuggingFace Inc. team. All rights reserved.
# ------------------------------------------------------------------------

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union,Callable

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm,  SwiGLUFFN
from rfdetr.models.backbone.rope_position_encoding import RopePositionEmbedding
from dinov3.layers.layer_scale import LayerScale
from rfdetr.models.backbone.attention import SelfAttention
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from transformers.utils.backbone_utils import BackboneMixin

from transformers.configuration_utils import PretrainedConfig
from transformers.utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/dinov3_with_registers-base"

# General docstring
_CONFIG_FOR_DOC = "WindowedDinov3WithRegistersConfig"

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

class WindowedDinov3WithRegistersConfig(BackboneConfigMixin, PretrainedConfig):

    model_type = "dinov3_with_registers"
    DYNAMIC_CLASS_MAPPING: Dict[str, Dict[str, Callable[..., nn.Module]]] = {
        "norm_layer": {
            "LayerNorm": nn.LayerNorm,
        },
        "attn_class": {
            "SelfAttention": SelfAttention,
        },
        "ffn_layer": {
            "Mlp": Mlp,
            "SwiGLUFFN": SwiGLUFFN, # hidden_features = d × 3 / 2
        },
        "act_layer": {
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "ReLU": nn.ReLU,
        }
    }
    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        proj_bias=True,
        layerscale_value=1.0,
        drop_path_rate=0.0,
        use_swiglu_ffn=False,
        num_register_tokens=4,
        out_features=None,
        out_indices=None,
        apply_layernorm=True,
        reshape_hidden_states=True,
        num_windows=1,
        window_block_indexes=None,
        gradient_checkpointing=False,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "fp32",
        dynamic_norm_layer: str | None = None,
        dynamic_attn_class: str | None = None,
        dynamic_ffn_layer: str | None = None,
        dynamic_act_layer: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_swiglu_ffn = use_swiglu_ffn
        self.num_register_tokens = num_register_tokens
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states
        self.num_windows = num_windows
        self.pos_embed_rope_base = pos_embed_rope_base
        self.pos_embed_rope_min_period = pos_embed_rope_min_period
        self.pos_embed_rope_max_period = pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords = pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords = pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords = pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords = pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        self.window_block_indexes = list(range(num_hidden_layers)) if window_block_indexes is None else window_block_indexes
        self.gradient_checkpointing = gradient_checkpointing

        if dynamic_norm_layer is not None:
            if dynamic_norm_layer not in self.DYNAMIC_CLASS_MAPPING["norm_layer"]:
                raise ValueError(f"不支持的归一化层：{dynamic_norm_layer}，可选值：{list(self.DYNAMIC_CLASS_MAPPING['norm_layer'].keys())}")
            self.norm_layer = self.DYNAMIC_CLASS_MAPPING["norm_layer"][dynamic_norm_layer]
        else:
            self.norm_layer = nn.LayerNorm

        if dynamic_attn_class is not None:
            if dynamic_attn_class not in self.DYNAMIC_CLASS_MAPPING["attn_class"]:
                raise ValueError(f"不支持的注意力层：{dynamic_attn_class}，可选值：{list(self.DYNAMIC_CLASS_MAPPING['attn_class'].keys())}")
            self.attn_class = self.DYNAMIC_CLASS_MAPPING["attn_class"][dynamic_attn_class]
        else:
            self.attn_class = SelfAttention  

        if dynamic_ffn_layer is not None:
            if dynamic_ffn_layer not in self.DYNAMIC_CLASS_MAPPING["ffn_layer"]:
                raise ValueError(f"不支持的FFN层：{dynamic_ffn_layer}，可选值：{list(self.DYNAMIC_CLASS_MAPPING['ffn_layer'].keys())}")
            self.ffn_layer = self.DYNAMIC_CLASS_MAPPING["ffn_layer"][dynamic_ffn_layer]
        else:
            self.ffn_layer = SwiGLUFFN if self.use_swiglu_ffn else Mlp

        if dynamic_act_layer is not None:
            if dynamic_act_layer not in self.DYNAMIC_CLASS_MAPPING["act_layer"]:
                raise ValueError(f"不支持的激活函数：{dynamic_act_layer}，可选值：{list(self.DYNAMIC_CLASS_MAPPING['act_layer'].keys())}")
            self.act_layer = self.DYNAMIC_CLASS_MAPPING["act_layer"][dynamic_act_layer]
        else:
            self.act_layer = nn.GELU

class Dinov3WithRegistersPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class WindowedDinov3WithRegistersEmbeddings(nn.Module):
    """
    Construct the CLS token, mask token, register tokens, position and patch embeddings.
    """

    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size)) if config.num_register_tokens > 0 else None
        self.patch_embeddings = Dinov3WithRegistersPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images. This implementation supports torch.jit tracing while maintaining backwards compatibility
        with the original implementation.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
        - https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
        """
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # Skip interpolation for matching dimensions (unless tracing)
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        # Handle class token and patch embeddings separately
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]

        # Calculate new dimensions
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        # Reshape for interpolation
        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Store original dtype for restoration after interpolation
        target_dtype = patch_pos_embed.dtype

        # Interpolate at float32 precision
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            size=(torch_int(height), torch_int(width)),  # Explicit size instead of scale_factor
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(dtype=target_dtype)

        # Validate output dimensions if not tracing
        if not torch.jit.is_tracing():
            if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
                raise ValueError("Width or height does not match with the interpolated position embeddings")

        # Reshape back to original format
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Combine class and patch embeddings
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        x = self.patch_embeddings(pixel_values.to(dtype=target_dtype))  # [B, N, C], N=H/ps*W/ps

        if bool_masked_pos is not None:
            x = torch.where(bool_masked_pos.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        if self.config.num_windows > 1:
            num_wins = self.config.num_windows
            win_grid_h = grid_h // num_wins
            win_grid_w = grid_w // num_wins
        else:
            num_wins = 1
            win_grid_h = grid_h
            win_grid_w = grid_w

        if num_wins > 1:
            # 按行优先进行左上到右下的划分
            cls_with_pos = x[:, :1]
            pix = x[:, 1:]  # [B, grid_h*grid_w, C]
            pix = pix.view(B, grid_h, grid_w, -1)
            pix = pix.view(B, num_wins, win_grid_h, num_wins, win_grid_w, -1)
            pix = pix.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, nw, nw, win_h, win_w, C]
            pix = pix.view(B * num_wins * num_wins, win_grid_h * win_grid_w, -1)
            cls_with_pos = cls_with_pos.repeat(num_wins * num_wins, 1, 1)
            x = torch.cat([cls_with_pos, pix], dim=1)

        # add register tokens
        if self.config.num_register_tokens > 0:
            x = torch.cat([x[:, :1], self.register_tokens.expand(x.size(0), -1, -1), x[:, 1:]], dim=1)
        x = self.dropout(x)

        return x, grid_h, grid_w, win_grid_h, win_grid_w


class Dinov3WithRegistersSelfAttention(nn.Module):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class Dinov3WithRegistersSdpaSelfAttention(Dinov3WithRegistersSelfAttention):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Dinov3WithRegistersModel is using Dinov3WithRegistersSdpaSelfAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class Dinov3WithRegistersSelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov3WithRegistersLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Dinov3WithRegistersAttention(nn.Module):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()
        self.attention = Dinov3WithRegistersSelfAttention(config)
        self.output = Dinov3WithRegistersSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class Dinov3WithRegistersSdpaAttention(Dinov3WithRegistersAttention):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__(config)
        self.attention = Dinov3WithRegistersSdpaSelfAttention(config)



def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


Dinov3_WITH_REGISTERS_ATTENTION_CLASSES = {
    "eager": Dinov3WithRegistersAttention,
    "sdpa": Dinov3WithRegistersSdpaAttention,
}


class WindowedDinov3WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()

        self.num_windows = config.num_windows
        norm_layer: Callable[..., nn.Module] = config.norm_layer if hasattr(config, 'norm_layer') else nn.LayerNorm
        attn_class: Callable[..., nn.Module] = config.attn_class if hasattr(config, 'attn_class') else SelfAttention
        ffn_layer: Callable[..., nn.Module] = config.ffn_layer if hasattr(config, 'ffn_layer') else Mlp
        act_layer: Callable[..., nn.Module] = config.act_layer if hasattr(config, 'act_layer') else nn.GELU
        self.num_register_tokens=config.num_register_tokens
        self.norm1 = norm_layer(config.hidden_size, eps=config.layer_norm_eps) # d
        self.attn = attn_class(
            config.hidden_size, 
            num_heads=config.num_attention_heads, 
            qkv_bias=config.qkv_bias,
            proj_bias=config.proj_bias,
            attn_drop=config.attention_probs_dropout_prob,
            proj_drop=config.attention_probs_dropout_prob,
            mask_k_bias=config.mask_k_bias,
        )

        self.ls1 = LayerScale(config.hidden_size)

        self.norm2 = norm_layer(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ffn_layer(
            config.hidden_size,
            hidden_features=config.mlp_ratio*config.hidden_size,
            act_layer=act_layer,
            drop=config.hidden_dropout_prob,
            bias=config.ffn_bias,
        )
        self.ls2 = LayerScale(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        run_full_attention: bool = False,
        orig_h: int = 0,
        orig_w: int = 0,
        win_h: int = 0,
        win_w: int = 0,
        rope_embed: nn.Module = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        assert head_mask is None, "head_mask is not supported for windowed attention"
        assert not output_attentions, "output_attentions is not supported for windowed attention"
        shortcut = hidden_states
        num_wins=self.num_windows
        if run_full_attention:
            # reshape x to remove windows
            B, HW, C = hidden_states.shape # HW: cls+reg+patch
            cls_per_window = 1
            reg_per_window = self.num_register_tokens
            patch_per_window = HW - cls_per_window - reg_per_window
            num_windows_squared = num_wins ** 2
            B_new = B // num_windows_squared
            windows = hidden_states.view(B_new, num_windows_squared , HW, C)
            all_cls = windows[:, :, 0:cls_per_window, :]
            all_reg = windows[:, :, cls_per_window:cls_per_window + reg_per_window, :]
            all_patch = windows[:, :, cls_per_window + reg_per_window:, :]
            flattened_cls = all_cls.reshape(B_new, num_windows_squared * cls_per_window, C)
            flattened_reg = all_reg.reshape(B_new, num_windows_squared * reg_per_window, C)
            flattened_patch = all_patch.reshape(B_new, num_windows_squared * patch_per_window, C)
            hidden_states = torch.cat([flattened_cls, flattened_reg, flattened_patch], dim=1)
            
            sin_full, cos_full = rope_embed(H=orig_h, W=orig_w)  # HW,D
            sin = sin_full.unsqueeze(0).repeat(B_new, 1, 1)  
            cos = cos_full.unsqueeze(0).repeat(B_new, 1, 1)
            rope_sincos = (sin, cos)
        else:
            full_rope_sincos = rope_embed(H=orig_h, W=orig_w)  # sin: [orig_HW, D], cos: [orig_HW, D]
            sin_full, cos_full = full_rope_sincos
            D = sin_full.shape[-1]

            sin_full_grid = sin_full.view(orig_h, orig_w, D)
            cos_full_grid = cos_full.view(orig_h, orig_w, D)

            win_patch_h = win_h
            win_patch_w = win_w
            sin_windows = []
            cos_windows = []

            for i in range(num_wins):  
                for j in range(num_wins):
                    h_start = i * win_patch_h
                    h_end = h_start + win_patch_h
                    w_start = j * win_patch_w
                    w_end = w_start + win_patch_w

                    sin_win = sin_full_grid[h_start:h_end, w_start:w_end, :]  # [win_patch_h, win_patch_w, D]
                    cos_win = cos_full_grid[h_start:h_end, w_start:w_end, :]

                    sin_win_flat = sin_win.flatten(0, 1)  # [win_patch_h*win_patch_w, D]
                    cos_win_flat = cos_win.flatten(0, 1)

                    sin_windows.append(sin_win_flat)
                    cos_windows.append(cos_win_flat)

            sin_stacked = torch.stack(sin_windows, dim=0)
            cos_stacked = torch.stack(cos_windows, dim=0)

            B_total = hidden_states.shape[0]
            B_original = B_total // (num_wins ** 2)

            sin_expanded = sin_stacked.unsqueeze(0).repeat(B_original, 1, 1, 1).flatten(0, 1)
            cos_expanded = cos_stacked.unsqueeze(0).repeat(B_original, 1, 1, 1).flatten(0, 1)
            rope_sincos = (sin_expanded, cos_expanded)
        
        self_attention_outputs = self.attn(
            self.norm1(hidden_states),  # in Dinov3WithRegisters, layernorm is applied before self-attention
            rope=rope_sincos,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0] # [B_total, N, C]

        if run_full_attention:
            # reshape x to add windows back
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            cls_per_window = 1
            reg_per_window = self.num_register_tokens
            window_seq_length = (HW // num_windows_squared) 
            patch_per_window = window_seq_length - cls_per_window - reg_per_window 
            total_cls = num_windows_squared * cls_per_window  
            total_reg = num_windows_squared * reg_per_window  

            all_cls = hidden_states[:, :total_cls, :].reshape(B, num_windows_squared, cls_per_window, C)
            all_reg = hidden_states[:, total_cls:total_cls+total_reg, :].reshape(B, num_windows_squared, reg_per_window, C)
            all_patch = hidden_states[:, total_cls+total_reg:, :].reshape(B, num_windows_squared, patch_per_window, C)
            
            windows = torch.cat([all_cls, all_reg, all_patch], dim=2)
            
            hidden_states = windows.reshape(B * num_windows_squared, window_seq_length, C)
            
            attention_output = attention_output.reshape(B, num_windows_squared, window_seq_length, C)
            attention_output = attention_output.reshape(B * num_windows_squared, window_seq_length, C)

        attention_output = self.ls1(attention_output)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + shortcut

        # in Dinov3WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.ls2(layer_output)

        # second residual connection
        layer_output = layer_output + hidden_states
        outputs = (layer_output,) + outputs
        return outputs

class WindowedDinov3WithRegistersEncoder(nn.Module):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([WindowedDinov3WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = config.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        orig_h: int,
        orig_w: int,
        win_h: int,
        win_w: int,
        rope_embed: nn.Module,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            if i > int(self.config.out_features[-1][5:]):
                # early stop if we have reached the last output feature
                break
            
            run_full_attention = i not in self.config.window_block_indexes

            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    run_full_attention,
                    orig_h,
                    orig_w,
                    win_h,
                    win_w,
                    rope_embed,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, run_full_attention, orig_h, orig_w, win_h, win_w, rope_embed)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class WindowedDinov3WithRegistersPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = WindowedDinov3WithRegistersConfig
    base_model_prefix = "Dinov3_with_registers"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov3WithRegistersSwiGLUFFN"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, WindowedDinov3WithRegistersEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


_EXPECTED_OUTPUT_SHAPE = [1, 257, 768]


Dinov3_WITH_REGISTERS_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Dinov3WithRegistersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

Dinov3_WITH_REGISTERS_BASE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Only relevant for
            pre-training.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Dinov3WithRegisters Model transformer outputting raw hidden-states without any specific head on top.",
    Dinov3_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov3WithRegistersModel(WindowedDinov3WithRegistersPreTrainedModel):
    def __init__(self, config: WindowedDinov3WithRegistersConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = WindowedDinov3WithRegistersEmbeddings(config)
        self.encoder = WindowedDinov3WithRegistersEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Dinov3WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(Dinov3_WITH_REGISTERS_BASE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/Dinov3_with_registers-small-imagenet1k-1-layer"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

Dinov3_WITH_REGISTERS_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`BitImageProcessor.preprocess`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    Dinov3WithRegisters Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    Dinov3_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov3WithRegistersForImageClassification(WindowedDinov3WithRegistersPreTrainedModel):
    def __init__(self, config: WindowedDinov3WithRegistersConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.Dinov3_with_registers = WindowedDinov3WithRegistersModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(Dinov3_WITH_REGISTERS_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.Dinov3_with_registers(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]

        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        logits = self.classifier(linear_input)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



@add_start_docstrings(
    """
    Dinov3WithRegisters backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    Dinov3_WITH_REGISTERS_START_DOCSTRING,
)
class WindowedDinov3WithRegistersBackbone(WindowedDinov3WithRegistersPreTrainedModel, BackboneMixin):
    def __init__(self, config: WindowedDinov3WithRegistersConfig):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        self.embeddings = WindowedDinov3WithRegistersEmbeddings(config)
        self.encoder = WindowedDinov3WithRegistersEncoder(config)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            base=config.pos_embed_rope_base,
            min_period=config.pos_embed_rope_min_period,
            max_period=config.pos_embed_rope_max_period,
            normalize_coords=config.pos_embed_rope_normalize_coords,
            shift_coords=config.pos_embed_rope_shift_coords,
            jitter_coords=config.pos_embed_rope_jitter_coords,
            rescale_coords=config.pos_embed_rope_rescale_coords,
            dtype=dtype_dict[config.pos_embed_rope_dtype],
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.num_register_tokens = config.num_register_tokens
        self.init()
        # Initialize weights and apply final processing
        self.post_init()

    def init(self): #TODO
        self.rope_embed._init_weights()

    def get_input_embeddings(self) -> Dinov3WithRegistersPatchEmbeddings:
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(Dinov3_WITH_REGISTERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor, # B C H W
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:
        Returns:

        Examples:


        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-with-registers-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # print("input:",pixel_values)
        embedding_output,orig_h,orig_w,win_h,win_w = self.embeddings(pixel_values) # 
        # print("embedding_output[0]",embedding_output[0])
        outputs = self.encoder(
            embedding_output, orig_h, orig_w, win_h, win_w, self.rope_embed, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # for i,hidden_state in enumerate(hidden_states):
        #     print(f"{i}:",hidden_state[0])
        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, self.num_register_tokens + 1 :]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    
                    B_img, _, H_img, W_img = pixel_values.shape
                    ps = self.config.patch_size
                    H_grid = H_img // ps
                    W_grid = W_img // ps

                    if self.config.num_windows > 1:
                        Wn = self.config.num_windows
                        W2 = Wn * Wn
                        h_win = H_grid // Wn
                        w_win = W_grid // Wn

                        # (B*W2, h_win*w_win, C) → (B, Wn, Wn, h_win, w_win, C)
                        B_all = hidden_state.shape[0]
                        assert B_all % W2 == 0, "batch 与窗口数不整除"
                        B = B_all // W2
                        C = hidden_state.shape[-1]
                        x = hidden_state.view(B, Wn, Wn, h_win, w_win, C)
                        # → (B, H_grid, W_grid, C)
                        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_grid, W_grid, C)
                    else:
                        # (B, H_grid*W_grid, C) → (B, H_grid, W_grid, C)
                        B = hidden_state.shape[0]
                        C = hidden_state.shape[-1]
                        x = hidden_state.view(B, H_grid, W_grid, C)

                    # → (B, C, H_grid, W_grid)
                    hidden_state = x.permute(0, 3, 1, 2).contiguous()
                
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )


__all__ = [
    "WindowedDinov3WithRegistersPreTrainedModel",
    "WindowedDinov3WithRegistersModel",
    "WindowedDinov3WithRegistersForImageClassification",
    "WindowedDinov3WithRegistersBackbone",
]