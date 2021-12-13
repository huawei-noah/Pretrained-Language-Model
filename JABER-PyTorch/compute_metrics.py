# coding=utf-8
# 2021.09.29 - changed the F1 average method from binary to macro
#              Huawei Technologies Co., Ltd. 
# Source: https://github.com/Alue-Benchmark/alue_baselines/blob/master/bert-baselines/compute_metrics.py


# Modified version of Transformers compute metrics script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/src/transformers/data/metrics/__init__.py

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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


try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import f1_score, accuracy_score, jaccard_score

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def acc_and_f1(preds, labels, average="macro"):
        f1 = f1_score(y_true=labels, y_pred=preds, average=average)
        acc = accuracy_score(preds, labels)

        return {
            "f1": f1,
            "acc": acc,
        }

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]

        return {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    def jaccard_and_f1(preds, labels):
        # jaccard = jaccard_score(y_true=labels, y_pred=preds, average="samples")
        jaccard = jaccard_score(y_true=labels, y_pred=preds, average="macro")
        f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
        f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")

        return {
            "jaccard": jaccard,
            "f1-macro": f1_macro,
            "f1-micro": f1_micro,
        }

    def alue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "mq2q":
            return acc_and_f1(preds, labels)
        elif task_name == "mdd":
            return acc_and_f1(preds, labels, average="macro")
        elif task_name == "fid":
            return acc_and_f1(preds, labels)
        elif task_name == "svreg":
            return pearson_and_spearman(preds, labels)
        elif task_name == "sec":
            return jaccard_and_f1(preds, labels)
        elif task_name == "oold":
            return acc_and_f1(preds, labels)
        elif task_name == "ohsd":
            return acc_and_f1(preds, labels)
        elif task_name == "xnli":
            return acc_and_f1(preds, labels, average="macro")       
        else:
            raise KeyError(task_name)
