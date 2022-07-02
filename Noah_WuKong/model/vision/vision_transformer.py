#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 2022.02.13 - Add TokenReduction layer before model outputs.
#     Huawei Technologies Co., Ltd.
# 2022.01.03 - A default value of argument `heads` is set in VisionTransformer.
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
import torch
from mmcv.cnn import kaiming_init, constant_init
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import MODELS
from ..language.transformer import LayerNorm, Transformer
from ..modules import TokenReduction


@MODELS.register_module()
class VisionTransformer(nn.Module):
    conv_stem_setting = dict(small=([24, 48, 96, 96, 192], (2, 2, 2, 2, 2)),
                             middle=([48, 96, 192, 384], (2, 2, 2, 2)),
                             large=([64, 128, 256, 512], (2, 2, 2, 2)))

    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            output_dim: int,
            init_scale: float = 1.,
            token_reduction=None
    ):
        super().__init__()
        heads = width // 64
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        scale = (width / init_scale) ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.num_tokens = (input_resolution // patch_size) ** 2 + 1
        self.num_tokens_side = input_resolution // patch_size
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_tokens, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        if token_reduction is not None:
            self.token_reduction = TokenReduction(
                in_channels=output_dim, **token_reduction)
        self.init_weights()

    @property
    def dtype(self):
        return self.positional_embedding.dtype

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x: torch.Tensor, return_full_embed=False):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        if not return_full_embed:
            x = self.ln_post(x[:, 0, :])
            if hasattr(self, "token_reduction"):
                raise NotImplementedError(
                    "Token-Reduction is only supported in token-wise similarity.")
        else:
            x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj

        if hasattr(self, "token_reduction"):
            cls_token, x = x[:, 0, :], x[:, 1:, :]
            x = x.view(-1, self.num_tokens_side, self.num_tokens_side, self.output_dim)
            x, weight = self.token_reduction(x)
            x = torch.cat([cls_token[:, None, :], x], dim=1)

        return x
