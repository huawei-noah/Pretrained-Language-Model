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
import logging
import os
from typing import Union

import torch
from mmcv import build_from_cfg
from mmcv.runner import load_state_dict
from torch import nn

from .builder import MODELS
from .language import TextTransformer
from .vision import SwinTransformer

logger = logging.getLogger(__name__)


@MODELS.register_module()
class Wukong(nn.Module):
    def __init__(self, embed_dim, visual: Union[SwinTransformer], text: TextTransformer,
                 is_token_wise=False, init_scale=1., pretrained=None):
        super().__init__()

        self.visual = build_from_cfg(
            visual, MODELS, default_args=dict(output_dim=embed_dim, init_scale=init_scale))
        self.transformer = build_from_cfg(
            text, MODELS, default_args=dict(output_dim=embed_dim, init_scale=init_scale))
        self.is_token_wise = is_token_wise
        if pretrained:
            self.load_pretrain(pretrained)

    @property
    def dtype(self):
        return self.visual.dtype

    def encode_image(self, image, return_full_embed=None):
        if return_full_embed is None:
            return_full_embed = self.is_token_wise
        return self.visual(image.type(self.dtype), return_full_embed=return_full_embed)

    def process_img_features(self, image_features):
        if self.is_token_wise:
            image_features = image_features if isinstance(self.visual, SwinTransformer) \
                else image_features[:, 1:, :]
        elif image_features.ndim == 3:
            image_features = image_features[:, 0, :]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, text, return_full_embed=None):
        if return_full_embed is None:
            return_full_embed = self.is_token_wise
        return self.transformer(text, return_full_embed=return_full_embed)

    def process_text_features(self, text_features, text):
        if self.is_token_wise:
            assert text_features.ndim == 3
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_pad_mask = text > 0
            text_features = text_features * text_pad_mask[:, :, None]
        else:
            if text_features.ndim == 3:
                text_features = text_features[
                    torch.arange(text_features.shape[0]), (text != 0).sum(dim=-1) - 1]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, image, text, label=None):
        pass

    def load_pretrain(self, pretrained):
        if not os.path.exists(pretrained):
            raise FileNotFoundError(f"File '{pretrained}' not exists")
        logger.info(f"Loading pretrained model from: {pretrained}")
        state_dict = torch.load(pretrained, map_location="cpu")
        state_dict = state_dict['state_dict']

        # replace some keys
        key_replaces = [("token_learner", "token_reduction")]
        for key in list(state_dict.keys()):
            for rep in key_replaces:
                if rep[0] in key:
                    state_dict[key.replace(*rep)] = state_dict.pop(key)
        # delete some keys
        for key in ["loss.logit_scale"]:
            if key in state_dict:
                del state_dict[key]

        load_state_dict(self, state_dict, True)
