#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2022.01.02 - Make changes for drop-path and layer-scale.
#     Huawei Technologies Co., Ltd.
# Copyright (c) 2022, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2021, OpenAI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict

import torch
from mmcv.cnn import constant_init
from timm.models.layers import DropPath
from torch import nn

from ..builder import MODELS
from ..modules import LayerNorm, QuickGELU
from ..utils import auto_grad_checkpoint


class ResidualAttentionBlock(nn.Module):
    detect_inf = False

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 sandwich_ln=False, layer_scale_init_values=None, drop_path=0.):
        super().__init__()

        self.sandwich_ln = sandwich_ln
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp = [("c_fc", nn.Linear(d_model, d_model * 4)),
               ("gelu", QuickGELU()),
               ("c_proj", nn.Linear(d_model * 4, d_model))]
        if sandwich_ln:
            mlp.append(("post_ln", LayerNorm(d_model)))
        self.mlp = nn.Sequential(OrderedDict(mlp))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.has_layer_scale = layer_scale_init_values is not None
        if self.has_layer_scale:
            self.gamma_1 = nn.Parameter(layer_scale_init_values * torch.ones(d_model), requires_grad=True)
            self.gamma_2 = nn.Parameter(layer_scale_init_values * torch.ones(d_model), requires_grad=True)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def fix_inf(self, x):
        if not self.detect_inf:
            return x
        # to enable fp16 training
        is_fp16 = x.dtype == torch.float16 or torch.is_autocast_enabled()
        if is_fp16 and torch.isinf(x).any():
            clamp_value = torch.finfo(torch.float16).max - 1000
            x = torch.clamp(x, min=-clamp_value, max=clamp_value)
        return x

    def forward(self, x: torch.Tensor):
        if self.has_layer_scale:
            x = self.fix_inf(x + self.drop_path(self.gamma_1 * self.attention(self.ln_1(x))))
            x = self.fix_inf(x + self.drop_path(self.gamma_2 * self.mlp(self.ln_2(x))))
        else:
            x = self.fix_inf(x + self.drop_path(self.attention(self.ln_1(x))))
            x = self.fix_inf(x + self.drop_path(self.mlp(self.ln_2(x))))
        return x


class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
            sandwich_ln=False, layer_scale=False, drop_path=0., init_scale=1.
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        layer_scale_value = self.get_layer_scale_value(layers) if layer_scale else None
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(
                width, heads, attn_mask, sandwich_ln, layer_scale_value, drop_path)
                for _ in range(layers)]
        )
        self.init_weights(init_scale)

    @staticmethod
    def get_layer_scale_value(depth):
        if depth <= 18:
            return 0.1
        else:
            return 1e-2

    def forward(self, x: torch.Tensor):
        return auto_grad_checkpoint(self.resblocks, x)

    def init_weights(self, init_scale=1.):
        factor = self.width / init_scale
        proj_std = (factor ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = factor ** -0.5
        fc_std = (2 * factor) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for m in self.modules():
            if isinstance(m, LayerNorm):
                constant_init(m, 1)


@MODELS.register_module()
class TextTransformer(Transformer):
    def __init__(self, context_length, vocab_size, output_dim, init_scale=1., **kwargs):
        super().__init__(attn_mask=self.build_attention_mask(context_length),
                         init_scale=init_scale, **kwargs)
        self.num_tokens = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, self.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(context_length, self.width)
        )
        self.ln_final = LayerNorm(self.width)

        self.output_dim = output_dim
        self.text_projection = nn.Parameter(torch.empty(self.width, output_dim))
        self.init_weights_(init_scale)

    @staticmethod
    def build_attention_mask(context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def init_weights_(self, init_scale=1.):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=(self.width / init_scale) ** -0.5)

    @property
    def dtype(self):
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text: torch.Tensor, return_full_embed=False):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = super().forward(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        if not return_full_embed:
            x = x[torch.arange(x.shape[0]), (text != 0).sum(dim=-1) - 1]

        x = x @ self.text_projection

        return x
