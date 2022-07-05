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
import argparse
import logging
import time

import mmcv
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.transforms import (
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
    Compose,
)

from data.datasets import ImageNetDataset
from model.builder import build_model
from model.utils import token_wise_similarity

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def img_transform(n_px):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = [
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda img: img.convert("RGB"),
        ToTensor(),
        Normalize(mean, std),
    ]
    return Compose(transform)


@torch.no_grad()
def get_text_features(model, dataset, prompts, num_classes):
    num_prompts = 1 if isinstance(prompts, str) else len(prompts)
    texts = dataset.get_text_labels(prompts).cuda()
    text_batch_size = 1024
    text_features = []

    for i in range((len(texts) // text_batch_size) + 1):
        text = texts[i * text_batch_size: (i + 1) * text_batch_size]
        if len(text):
            text_features_ = model.encode_text(text)
            text_features_ = model.process_text_features(text_features_, text)
            text_features.append(text_features_)
    text_features = torch.cat(text_features)
    # prompt ensemble
    if num_prompts > 1:
        text_features = text_features.reshape(
            num_classes, num_prompts, *text_features.shape[1:]
        )
        if not model.is_token_wise:
            text_features = text_features.mean(1)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_logits(image_features, text_features, is_token_wise=False):
    logits = (
        token_wise_similarity(image_features, text_features).softmax(dim=-1)
        if is_token_wise
        else (image_features @ text_features.T).softmax(dim=-1)
    )
    return logits


def calculate_accuracy(logits, targets):
    k = min(5, dataset.num_classes)
    pred = logits.topk(k, 1, True, True)[1].t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred)).contiguous()
    return correct[:1].sum(0), correct[:k].sum(0)


@torch.no_grad()
def eval(dataloader, model, prompt="{}", eval_ensemble_prompts=True):
    dataset = dataloader.dataset
    num_classes = dataset.num_classes
    ensemble_prompts = dataset.prompts
    eval_ensemble_prompts = eval_ensemble_prompts and len(ensemble_prompts) > 1
    text_features = get_text_features(model, dataset, prompt, num_classes)
    text_features_ensemble = (
        get_text_features(model, dataset, ensemble_prompts, num_classes)
        if eval_ensemble_prompts
        else None
    )

    results = dict()
    for idx, data in enumerate(dataloader):
        imgs, targets = map(lambda x: x.cuda(), data)
        image_features = model.encode_image(imgs)
        image_features = model.process_img_features(image_features)
        logits = get_logits(image_features, text_features, model.is_token_wise)
        # calculate accuracy
        acc1, acc5 = calculate_accuracy(logits, targets)
        results.setdefault("Top1_acc", []).append(acc1)
        results.setdefault("Top5_acc", []).append(acc5)
        if eval_ensemble_prompts:
            if not model.is_token_wise:
                logits = get_logits(
                    image_features, text_features_ensemble, model.is_token_wise
                )
            else:
                all_logits = []
                for i in range(text_features_ensemble.shape[1]):
                    text_feat = text_features_ensemble[:, i, ...]
                    all_logits.append(
                        get_logits(image_features, text_feat, model.is_token_wise)
                    )
                logits = sum(all_logits) / len(all_logits)
            acc1, acc5 = calculate_accuracy(logits, targets)
            results.setdefault("Top1_acc_ensemble", []).append(acc1)
            results.setdefault("Top5_acc_ensemble", []).append(acc5)
    for key, acc in results.items():
        acc = torch.cat(acc)
        acc = acc.float().mean().item() * 100
        results[key] = acc
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Wukong Arguments")
    parser.add_argument(
        "--config", type=str, required=True, help="Model config file.")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Model checkpoint file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="ImageNet data directory (.e.g, 'ILSVRC').",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size of data loading."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # parse config file
    logger.info(f"Loading config from: {args.config}")
    config = mmcv.Config.fromfile(args.config)
    config.model.pretrained = args.checkpoint
    # build model
    logger.info(f"Building model for evaluation.")
    model = build_model(config.model).cuda()
    model.eval()
    input_resolution = model.visual.input_resolution
    transform = img_transform(model.visual.input_resolution)
    # build dataset
    logger.info(f"Building dataset from: {args.data_dir}")
    dataset = ImageNetDataset(root_dir=args.data_dir, split="val", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=16,
    )
    # start evaluation
    logger.info(f"Staring evaluation.")
    time_s = time.time()
    results = eval(dataloader, model)
    # show results
    log_info = [f"{k}: {v:.3f}" for k, v in results.items()]
    logger.info(f"[Results] {'; '.join(log_info)}")
    logger.info(f"Evaluation done.")
