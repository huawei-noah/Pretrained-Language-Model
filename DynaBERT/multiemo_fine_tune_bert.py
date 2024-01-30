# coding=utf-8
# 2019.12.2-Changed for TinyBERT task-specific distillation
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

from sklearn.metrics import classification_report

from utils import result_to_text_file, dictionary_to_json
from transformers.modeling_bert import BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers.data.metrics import multiemo_compute_metrics as compute_metrics
from transformers.data.processors.multiemo import multiemo_convert_examples_to_features as convert_examples_to_features, \
    MultiemoProcessor
from transformers import AdamW, WarmupLinearSchedule

from transformers.file_utils import WEIGHTS_NAME, CONFIG_NAME

csv.field_size_limit(sys.maxsize)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return tensor_data, all_label_ids


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    all_logits = None

    for batch in tqdm(eval_dataloader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if output_mode == "regression":
        all_logits = np.squeeze(all_logits)
    result = compute_metrics(task_name, all_logits, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result, all_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--pretrained_model",
                        default=None,
                        type=str,
                        help="The pretrained model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay', '--wd',
                        default=0.01,
                        type=float,
                        metavar='W',
                        help='weight decay')
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    # added arguments
    parser.add_argument('--aug_train',
                        action='store_true')
    parser.add_argument('--eval_step',
                        type=int,
                        default=50)

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # intermediate distillation default parameters
    default_params = {
        "multiemo": {"num_train_epochs": 3, "max_seq_length": 128},
    }
    acc_tasks = ["multiemo"]

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # Prepare task settings
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    if task_name in default_params:
        args.max_seq_len = default_params[task_name]["max_seq_length"]

    if not args.do_eval:
        if task_name in default_params:
            args.num_train_epoch = default_params[task_name]["num_train_epochs"]

    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        raise ValueError("Task not found: %s" % task_name)

    if 'multiemo' in task_name:
        output_mode = 'classification'
    else:
        raise ValueError("Task not found: %s" % task_name)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model, do_lower_case=args.do_lower_case)

    if not args.do_eval:
        if not args.aug_train:
            train_examples = processor.get_train_examples(args.data_dir)
        else:
            train_examples = processor.get_aug_examples(args.data_dir)

        t_total = len(train_examples) // args.gradient_accumulation_steps * args.num_train_epochs

        train_features = convert_examples_to_features(
            train_examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        train_data, _ = get_tensor_data(output_mode, train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = BertForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=num_labels)
    model.to(device)
    if args.do_eval:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        result, _ = do_eval(model, task_name, eval_dataloader,
                            device, output_mode, eval_labels, num_labels)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    else:
        training_start_time = time.monotonic()

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        optimizer, scheduler = get_optimizer_and_scheduler(args, model, t_total)

        # Train and evaluate
        global_step = 0
        best_dev_acc = 0.0
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        for epoch_ in range(int(args.num_train_epochs)):
            tr_loss = 0.
            tr_cls_loss = 0.

            model.train()
            nb_tr_steps = 0

            for step, batch in enumerate(tqdm(train_dataloader, f"Epoch {epoch_ + 1}: ", ascii=True)):
                batch = tuple(t.to(device) for t in batch)
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                          'token_type_ids': batch[2] if args.model_type in ['bert'] else None}

                cls_loss = model(**inputs)[0]

                loss = cls_loss
                tr_cls_loss += cls_loss.item()

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                loss.backward()
                tr_loss += loss.item()
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            logger.info("***** Running evaluation *****")
            logger.info("  Epoch = {} iter {} step".format(epoch_, global_step))
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()

            loss = tr_loss / nb_tr_steps
            cls_loss = tr_cls_loss / nb_tr_steps

            result, _ = do_eval(model, task_name, eval_dataloader,
                                device, output_mode, eval_labels, num_labels)
            result['epoch'] = epoch_ + 1
            result['global_step'] = global_step
            result['cls_loss'] = cls_loss
            result['loss'] = loss
            result_to_text_file(result, output_eval_file)

            save_model = False

            if result['acc'] > best_dev_acc:
                best_dev_acc = result['acc']
                save_model = True

            if save_model:
                logger.info("***** Save model *****")
                model_to_save = model.module if hasattr(model, 'module') else model

                model_name = WEIGHTS_NAME
                output_model_file = os.path.join(args.output_dir, model_name)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

            model.train()

        # Measure End Time
        training_end_time = time.monotonic()

        diff = timedelta(seconds=training_end_time - training_start_time)
        diff_seconds = diff.total_seconds()

        training_parameters = vars(args)
        training_parameters['training_time'] = diff_seconds

        output_training_params_file = os.path.join(args.output_dir, "training_params.json")
        dictionary_to_json(training_parameters, output_training_params_file)

        #########################
        #       Test model      #
        #########################
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )

        test_data, test_labels = get_tensor_data(output_mode, test_features)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

        logger.info("\n***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_start_time = time.monotonic()
        model.eval()
        result, y_logits = do_eval(model, task_name, test_dataloader,
                                   device, output_mode, test_labels, num_labels)
        eval_end_time = time.monotonic()

        diff = timedelta(seconds=eval_end_time - eval_start_time)
        diff_seconds = diff.total_seconds()
        result['eval_time'] = diff_seconds
        result_to_text_file(result, os.path.join(args.output_dir, "test_results.txt"))

        y_pred = np.argmax(y_logits, axis=1)
        print('\n\t**** Classification report ****\n')
        print(classification_report(test_labels.numpy(), y_pred, target_names=label_list))

        report = classification_report(test_labels.numpy(), y_pred, target_names=label_list, output_dict=True)
        report['eval_time'] = diff_seconds
        dictionary_to_json(report, os.path.join(args.output_dir, "test_results.json"))


def get_optimizer_and_scheduler(args, model, t_total):
    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer, scheduler


if __name__ == "__main__":
    main()
