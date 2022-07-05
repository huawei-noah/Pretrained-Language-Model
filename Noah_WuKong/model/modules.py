#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, Huawei Technologies Co., Ltd. All rights reserved.
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
import torch.distributed as dist
import torch.nn as nn
from mmcv.cnn import kaiming_init
from torch.nn.functional import linear

from .utils import get_rank, get_world_size, is_distributed


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(torch.jit.ScriptModule):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if is_distributed():
            output = [torch.zeros_like(input) for _ in range(get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [input]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grads = torch.stack(grads)
        if is_distributed():
            dist.all_reduce(grads)
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out


class TokenReduction(nn.Module):
    def __init__(self, in_channels, num_tokens, num_groups=8, dropout_rate=.0):
        super(TokenReduction, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        self.num_groups = num_groups
        self.norm = LayerNorm(self.in_channels)

        self.attention_maps = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels, kernel_size=(1, 1),
                stride=(1, 1), padding=0, groups=self.num_groups, bias=False),
            nn.Conv2d(
                self.in_channels, self.num_tokens, kernel_size=(1, 1),
                stride=(1, 1), padding=0, bias=False),
        )
        self.feat_conv = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=0, groups=self.num_groups, bias=False
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, inputs):
        feature_shape = inputs.shape

        selected = inputs
        selected = self.norm(selected)
        selected = selected.permute(0, 3, 1, 2)
        selected = self.attention_maps(selected)
        selected = selected.permute(0, 2, 3, 1)
        selected = selected.contiguous().view(
            feature_shape[0], feature_shape[1] * feature_shape[2], -1)
        selected = selected.permute(0, 2, 1)
        selected = nn.functional.softmax(selected, dim=-1)

        # Reshape the input to align it with the output of the conv block.
        feat = inputs
        feat = feat.permute(0, 3, 1, 2)
        feat = self.feat_conv(feat)
        feat = feat.permute(0, 2, 3, 1)
        feat = feat.contiguous().view(feature_shape[0], feature_shape[1] * feature_shape[2], -1)

        # Produced the attended inputs.
        outputs = torch.einsum("...si,...id->...sd", selected, feat)
        outputs = self.dropout(outputs)

        return outputs, selected
