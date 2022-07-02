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
model = dict(
    type='Wukong',
    pretrained='',
    embed_dim=768,
    visual=dict(
        type='SwinTransformer',
        name='swin_large_patch4_window7_224',
        input_resolution=224,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        patch_size=4,
        num_classes=21841),
    text=dict(
        type='TextTransformer',
        context_length=32,
        vocab_size=21128,
        width=768,
        heads=12,
        layers=12),
    is_token_wise=False
)