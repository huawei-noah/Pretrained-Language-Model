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
import json
import logging
import os
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader, make_dataset, is_image_file

from .tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, split="val", transform=None):
        self.root = root_dir
        self.transform = transform
        self.split = split
        self.loader = pil_loader
        self.tokenizer = SimpleTokenizer()
        self.split_folder = dict(train='Data/CLS-LOC/train', val='Data/CLS-LOC/val')
        self.data, self.targets, self.classes = self.get_data()
        self.prompts = self.get_prompts(Path(__file__).parent.joinpath("res/prompts.txt"))

    def get_data(self):
        folder = os.path.join(self.root, self.split_folder[self.split])
        classes, class_to_idx = self._find_classes(folder)
        samples = make_dataset(folder, class_to_idx, is_valid_file=is_image_file)
        data, targets = zip(*samples)
        logger.info(f"Dataset summary: #examples={len(data)}; #classes={len(classes)}")
        return data, targets, classes

    @staticmethod
    def get_prompts(prompts_file):
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [x.strip() for x in f.readlines()]
        return lines

    def get_text_labels(self, text_templates: Union[str, list] = "{}"):
        if isinstance(text_templates, str):
            text_templates = [text_templates]
        text_labels = []
        for c in self.class_names:
            label = map(lambda x: x.replace("{}", c), text_templates)
            text_labels.extend(label)
        token = self.tokenizer.tokenize(text_labels)
        return token

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        if isinstance(data, str):
            data = self.loader(data)
        data_result = data if self.transform is None else self.transform(data)
        return data_result, target

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _find_classes(root_dir):
        classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def class_names(self):
        classname_file = Path(__file__).parent.joinpath("res/classnames.json")
        with open(classname_file) as f:
            mapping = json.load(f)
        return [mapping[cls] for cls in self.classes]
