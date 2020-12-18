# coding=utf-8
# 2019.12.2-Changed for TinyBERT general distillation
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

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler,Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss

from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling import TinyBertForPreTraining, BertModel
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam

csv.field_size_limit(sys.maxsize)

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    if len(tokens) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(tokens)))
        logger.info('tokens: {}'.format(tokens))
        tokens = tokens[:max_seq_length]

    if len(tokens) != len(segment_ids):
        logger.info('tokens: {}\nsegment_ids: {}'.format(tokens, segment_ids))
        segment_ids = [0] * len(tokens)

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path('/cache')
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)

        logging.info("Loading training examples for epoch {}".format(epoch))

        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next

        # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(int(self.is_nexts[item])))


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-4,
                        type=float, metavar='W',
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
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')

    # Additional arguments
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)

    # This is used for running on Huawei Cloud.
    parser.add_argument('--data_url',
                        type=str,
                        default="")

    args = parser.parse_args()
    logger.info('args:{}'.format(args))

    samples_per_epoch = []
    for i in range(int(args.num_train_epochs)):
        epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
        metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print("Warning! There are fewer epochs of pregenerated data ({}) than training epochs ({}).".format(i, args.num_train_epochs))
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.num_train_epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(int(args.num_train_epochs)):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.continue_train:
        student_model = TinyBertForPreTraining.from_pretrained(args.student_model)
    else:
        student_model = TinyBertForPreTraining.from_scratch(args.student_model)
    teacher_model = BertModel.from_pretrained(args.teacher_model)

    # student_model = TinyBertForPreTraining.from_scratch(args.student_model, fit_size=teacher_model.config.hidden_size)
    student_model.to(device)
    teacher_model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        teacher_model = DDP(teacher_model)
    elif n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    size = 0
    for n, p in student_model.named_parameters():
        logger.info('n: {}'.format(n))
        logger.info('p: {}'.format(p.nelement()))
        size += p.nelement()

    logger.info('Total parameters: {}'.format(size))

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    loss_mse = MSELoss()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info("  Num examples = {}".format(total_train_examples))
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        student_model.train()
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                if input_ids.size()[0] != args.train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.

                student_atts, student_reps = student_model(input_ids, segment_ids, input_mask)
                teacher_reps, teacher_atts, _ = teacher_model(input_ids, segment_ids, input_mask)
                teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
                teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]

                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    att_loss += loss_mse(student_att, teacher_att)

                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
                new_student_reps = student_reps

                for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
                    rep_loss += loss_mse(student_rep, teacher_rep)

                loss = att_loss + rep_loss

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)

                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / nb_tr_steps

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (global_step + 1) % args.eval_step == 0:
                        result = {}
                        result['global_step'] = global_step
                        result['loss'] = mean_loss
                        result['att_loss'] = mean_att_loss
                        result['rep_loss'] = mean_rep_loss
                        output_eval_file = os.path.join(args.output_dir, "log.txt")
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            for key in sorted(result.keys()):
                                logger.info("  %s = %s", key, str(result[key]))
                                writer.write("%s = %s\n" % (key, str(result[key])))

                        # Save a trained model
                        model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                        logging.info("** ** * Saving fine-tuned model ** ** * ")
                        # Only save the model it-self
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                        output_model_file = os.path.join(args.output_dir, model_name)
                        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(args.output_dir)

                        if oncloud:
                            logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                            logging.info(mox.file.list_directory('.', recursive=True))
                            mox.file.copy_parallel(args.output_dir, args.data_url)
                            mox.file.copy_parallel('.', args.data_url)

            model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

            output_model_file = os.path.join(args.output_dir, model_name)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)

            if oncloud:
                logging.info(mox.file.list_directory(args.output_dir, recursive=True))
                logging.info(mox.file.list_directory('.', recursive=True))
                mox.file.copy_parallel(args.output_dir, args.data_url)
                mox.file.copy_parallel('.', args.data_url)


if __name__ == "__main__":
    main()
