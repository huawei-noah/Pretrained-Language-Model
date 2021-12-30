# coding: utf-8
# 2021.12.30-Changed for pretraining
#      Huawei Technologies Co., Ltd. <yinyichun@huawei.com>
# Copyright 2021 Huawei Technologies Co., Ltd.
# Copyright 2019 Sinovation Ventures AI Institute
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

import os

from argparse import ArgumentParser
from pathlib import Path

import json
import random
import numpy as np
from collections import namedtuple
import time
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import pickle
import collections

# This is used for running on Huawei Cloud.
oncloud = True
try:
    import moxing as mox
except:
    oncloud = False

from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_super_kd import SuperTinyBertForPreTraining, SuperBertForPreTraining, BertConfig
from transformer.modeling_base import BertModel
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam
from utils import sample_arch_4_kd, sample_arch_4_mlm

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import logging

from apex.parallel import DistributedDataParallel as DDP

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask lm_label_ids")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    mask_indices = sorted(random.sample(range(1, len(tokens)-1), num_to_mask))
    masked_token_labels = [tokens[index] for index in mask_indices]
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = random.choice(vocab_list)
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def convert_example_to_features(example, args):
    tokens1 = example["tokens"]
    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(tokens1,
                                                                                 args.masked_lm_prob,
                                                                                 args.max_predictions_per_seq,
                                                                                 args.vocab_list)

    assert len(tokens) <= args.max_seq_length  # The preprocessed data should be already truncated
    try:
        input_ids = args.tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = args.tokenizer.convert_tokens_to_ids(masked_lm_labels)
    except Exception as e:
        print(e)
        print(tokens1)
        print(tokens)
        print(masked_lm_labels)

    input_array = np.zeros(args.max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(args.max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    lm_label_array = np.full(args.max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             # segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
    )
    return features


def mask_and_choose(batch, num_samples, args):
    seq_len = args.max_seq_length
    input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
    input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
    lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)

    for i, line in enumerate(batch):
        example = json.loads(line)
        features = convert_example_to_features(example, args)
        input_ids[i] = features.input_ids
        input_masks[i] = features.input_mask
        lm_label_ids[i] = features.lm_label_ids

    input_ids = torch.from_numpy(input_ids.astype(np.int64))
    input_masks = torch.from_numpy(input_masks.astype(np.int64))
    lm_label_ids = torch.from_numpy(lm_label_ids.astype(np.int64))
    return (input_ids, input_masks, lm_label_ids)


def run_task(data_files, args):
    start_time = time.time()
    logging.info('Running thread %s, %s files', args.rank, len(data_files))
    for i, data_file in enumerate(data_files):
        input_data_file = os.path.join(args.pregenerated_data, data_file)
        logging.info("Loading inputs from file %s", input_data_file)
        examples = []
        with Path(input_data_file).open() as f:
            for j, line in enumerate(tqdm(f, desc="Training examples")):
                examples.append(line.strip())
        if args.debug:
            examples = examples[:20000]  # debug

        logging.info("num_samples before cut %s", len(examples))
        num_samples = int(len(examples)/args.world_size)
        input_ids, input_masks, lm_label_ids = mask_and_choose(examples[args.rank*num_samples:(args.rank+1)*num_samples], num_samples, args)
        logging.info("num_samples after cut %s", num_samples)

        # for k in range(int(hvd.size())):
        data_file_cached = os.path.join(args.local_data_dir, data_file + '.cached.' + str(args.rank))
        logging.info("cached file %s", data_file_cached)
        with open(data_file_cached, "wb") as handle:
            pickle.dump([input_ids, input_masks, lm_label_ids], handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info('%s/%s processed in thread %s, time cost is %.2f secs' % (i + 1, len(data_files), args.rank, time.time() - start_time))


def load_doc_tokens_ngrams(args):
    data_files = []
    for inputfile in os.listdir(args.pregenerated_data):
        input_file = os.path.join(args.pregenerated_data, inputfile)
        if os.path.isfile(input_file) and inputfile.endswith('json') and inputfile.startswith('train_doc_tokens_ngrams'):
            data_files.append(inputfile)

    file_count = len(data_files)

    run_task(data_files, args)

    t_input_ids, t_input_masks, t_lm_label_ids, t_ngram_ids, t_ngram_masks, t_ngram_starts, t_ngram_ends = [], [], [], [], [], [], []
    for i in range(file_count):
        data_file_cached = os.path.join(args.local_data_dir, data_files[i] + '.cached.' + str(args.rank))
        with open(data_file_cached, "rb") as handle:
            input_ids, input_masks, lm_label_ids = pickle.load(handle)

        logging.info("Loading inputs from cached file %s", data_file_cached)
        logging.info("num_samples %s", len(input_ids))

        if i == 0:
            t_input_ids, t_input_masks, t_lm_label_ids = [input_ids], [input_masks], [lm_label_ids]
        else:
            t_input_ids.append(input_ids)
            t_input_masks.append(input_masks)
            t_lm_label_ids.append(lm_label_ids)
        logger.info("Dataset %s loaded", data_file_cached)
    t_input_ids = torch.cat(t_input_ids, 0)
    t_input_masks = torch.cat(t_input_masks, 0)
    t_lm_label_ids = torch.cat(t_lm_label_ids, 0)
    logging.info("total num_samples %s", len(t_input_ids))
    for i in range(1):
        logging.info("*** Example ***")
        logging.info("block %s" % i)
        tokens = args.tokenizer.convert_ids_to_tokens(t_input_ids[i].tolist())
        logging.info("inputs: %s" % ' '.join([str(item) for item in tokens]))
        logging.info("input_masks: %s" % ' '.join([str(item) for item in t_input_masks[i].tolist()]))
        logging.info("lm_label_ids: %s" % ' '.join([str(item) for item in t_lm_label_ids[i].tolist()]))

    dataset = TensorDataset(t_input_ids, t_input_masks, t_lm_label_ids)

    return dataset


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=str, required=True, default='/nas/hebin/data/english-exp/books_wiki_tokens_ngrams')
    parser.add_argument('--s3_output_dir', type=str, default='huawei_yun')
    parser.add_argument('--student_model', type=str, default='8layer_bert', required=True)
    parser.add_argument('--teacher_model', type=str, default='electra_base')
    parser.add_argument('--cache_dir', type=str, default='/cache', help='')

    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--scratch',
                        action='store_true',
                        help="Whether to train from scratch")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Whether to debug")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--already_trained_epoch",
                        default=0,
                        type=int)
    parser.add_argument("--masked_lm_prob", type=float, default=0.0,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=77,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--warmup_steps", default=10000, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_workers", type=int, default=4, help="num_workers.")
    parser.add_argument("--continue_index", type=int, default=0, help="")
    parser.add_argument("--threads", type=int, default=27,
                        help="Number of threads to preprocess input data")

    # Search space for sub_bart architecture
    parser.add_argument('--layer_num_space', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--hidden_size_space', nargs='+', type=int, default=[128, 768])
    parser.add_argument('--qkv_size_space', nargs='+', type=int, default=[180, 768])
    parser.add_argument('--intermediate_size_space', nargs='+', type=int, default=[128, 3072])
    parser.add_argument('--head_num_space', nargs='+', type=int, default=[1, 12])
    parser.add_argument('--sample_times_per_batch', type=int, default=1)
    parser.add_argument('--further_train', action='store_true')
    parser.add_argument('--mlm_loss', action='store_true')

    # Argument for Huawei yun
    parser.add_argument('--data_url', type=str, default='', help='s3 url')
    parser.add_argument("--train_url", type=str, default="", help="s3 url")

    args = parser.parse_args()

    assert (torch.cuda.is_available())
    device_count = torch.cuda.device_count()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Call the init process
    # init_method = 'tcp://'
    init_method = ''
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port

    # Manually set the device ids.
    # if device_count > 0:
    # args.local_rank = args.rank % device_count
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print('device_id: %s' % args.local_rank)
    print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    print(init_method)

    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size,
                                         rank=args.rank, init_method=init_method)

    LOCAL_DIR = args.cache_dir
    if oncloud:
        assert mox.file.exists(LOCAL_DIR)

    if args.local_rank == 0 and oncloud:
        logging.info(mox.file.list_directory(args.pregenerated_data, recursive=True))
        logging.info(mox.file.list_directory(args.student_model, recursive=True))

    local_save_dir = os.path.join(LOCAL_DIR, 'output', 'superbert', 'checkpoints')
    local_tsbd_dir = os.path.join(LOCAL_DIR, 'output', 'superbert', 'tensorboard')
    save_name = '_'.join([
        'superbert',
        'epoch', str(args.epochs),
        'lr', str(args.learning_rate),
        'bsz', str(args.train_batch_size),
        'grad_accu', str(args.gradient_accumulation_steps),
        str(args.max_seq_length),
        'gpu', str(args.world_size),
    ])
    bash_save_dir = os.path.join(local_save_dir, save_name)
    bash_tsbd_dir = os.path.join(local_tsbd_dir, save_name)
    if args.local_rank == 0:
        if not os.path.exists(bash_save_dir):
            os.makedirs(bash_save_dir)
            logger.info(bash_save_dir + ' created!')
        if not os.path.exists(bash_tsbd_dir):
            os.makedirs(bash_tsbd_dir)
            logger.info(bash_tsbd_dir + ' created!')

    local_data_dir_tmp = '/cache/data/tmp/'
    local_data_dir = local_data_dir_tmp + save_name

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.tokenizer = BertTokenizer.from_pretrained(args.student_model, do_lower_case=args.do_lower_case)
    args.vocab_list = list(args.tokenizer.vocab.keys())

    config = BertConfig.from_pretrained(os.path.join(args.student_model, CONFIG_NAME))
    logger.info("Model config {}".format(config))

    if args.further_train:
        if args.mlm_loss:
            student_model = SuperBertForPreTraining.from_pretrained(args.student_model, config)
        else:
            student_model = SuperTinyBertForPreTraining.from_pretrained(args.student_model, config)
    else:
        if args.mlm_loss:
            student_model = SuperBertForPreTraining.from_scratch(args.student_model, config)
        else:
            student_model = SuperTinyBertForPreTraining.from_scratch(args.student_model, config)

    student_model.to(device)

    if not args.mlm_loss:
        teacher_model = BertModel.from_pretrained(args.teacher_model)
        teacher_model.to(device)

    # build arch space
    min_hidden_size, max_hidden_size = args.hidden_size_space
    min_ffn_size, max_ffn_size = args.intermediate_size_space
    min_qkv_size, max_qkv_size = args.qkv_size_space
    min_head_num, max_head_num = args.head_num_space

    hidden_step = 4
    ffn_step = 4
    qkv_step = 12
    head_step = 1

    number_hidden_step = int((max_hidden_size - min_hidden_size) / hidden_step)
    number_ffn_step = int((max_ffn_size - min_ffn_size) / ffn_step)
    number_qkv_step = int((max_qkv_size - min_qkv_size) / qkv_step)
    number_head_step = int((max_head_num - min_head_num) / head_step)

    layer_numbers = list(range(args.layer_num_space[0], args.layer_num_space[1] + 1))
    hidden_sizes = [i * hidden_step + min_hidden_size for i in range(number_hidden_step + 1)]
    ffn_sizes = [i * ffn_step + min_ffn_size for i in range(number_ffn_step + 1)]
    qkv_sizes = [i * qkv_step + min_qkv_size for i in range(number_qkv_step + 1)]
    head_numbers = [i * head_step + min_head_num for i in range(number_head_step + 1)]

    ######
    if args.local_rank == 0:
        tb_writer = SummaryWriter(bash_tsbd_dir)

    global_step = 0
    step = 0
    tr_loss, tr_rep_loss, tr_att_loss = 0.0, 0.0, 0.0
    logging_loss, rep_logging_loss, att_logging_loss = 0.0, 0.0, 0.0
    end_time, start_time = 0, 0

    submodel_config = dict()

    if args.further_train:
        submodel_config['sample_layer_num'] = config.num_hidden_layers
        submodel_config['sample_hidden_size'] = config.hidden_size
        submodel_config['sample_intermediate_sizes'] = config.num_hidden_layers * [config.intermediate_size]
        submodel_config['sample_num_attention_heads'] = config.num_hidden_layers * [config.num_attention_heads]
        submodel_config['sample_qkv_sizes'] = config.num_hidden_layers * [config.qkv_size]

    for epoch in range(args.epochs):
        if epoch < args.continue_index:
            args.warmup_steps = 0
            continue

        args.local_data_dir = os.path.join(local_data_dir, str(epoch))
        if args.local_rank == 0:
            os.makedirs(args.local_data_dir)
        while 1:
            if os.path.exists(args.local_data_dir):
                epoch_dataset = load_doc_tokens_ngrams(args)
                break

        if args.local_rank == 0 and oncloud:
            logging.info('Dataset in epoch %s', epoch)
            logging.info(mox.file.list_directory(args.local_data_dir, recursive=True))

        train_sampler = DistributedSampler(epoch_dataset, num_replicas=1, rank=0)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        step_in_each_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        num_train_optimization_steps = step_in_each_epoch * args.epochs
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(epoch_dataset) * args.world_size)
        logger.info("  Num Epochs = %d", args.epochs)
        logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                     args.train_batch_size * args.gradient_accumulation_steps * args.world_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Num steps = %d", num_train_optimization_steps)

        if epoch == args.continue_index:
            # Prepare optimizer
            param_optimizer = list(student_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            warm_up_ratio = args.warmup_steps / num_train_optimization_steps
            print('warm_up_ratio: {}'.format(warm_up_ratio))
            optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate,
                                 e=args.adam_epsilon, schedule='warmup_linear',
                                 t_total=num_train_optimization_steps,
                                 warmup=warm_up_ratio)

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      " to use fp16 training.")
                student_model, optimizer = amp.initialize(student_model, optimizer,
                                                          opt_level=args.fp16_opt_level,
                                                          min_loss_scale=1) #

            # apex
            student_model = DDP(student_model, message_size=10000000,
                                gradient_predivide_factor=torch.distributed.get_world_size(),
                                delay_allreduce=True)

            if not args.mlm_loss:
                teacher_model = DDP(teacher_model, message_size=10000000,
                                    gradient_predivide_factor=torch.distributed.get_world_size(),
                                    delay_allreduce=True)
                teacher_model.eval()

            logger.info('apex data paralleled!')

        from torch.nn import MSELoss
        loss_mse = MSELoss()

        student_model.train()
        for step_, batch in enumerate(train_dataloader):
            step += 1
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, lm_label_ids = batch

            if not args.mlm_loss:
                teacher_last_rep, teacher_last_att = teacher_model(input_ids, input_masks)
                teacher_last_att = torch.where(teacher_last_att <= -1e2,
                                               torch.zeros_like(teacher_last_att).to(device),
                                               teacher_last_att)
                teacher_last_rep.detach()
                teacher_last_att.detach()

            for sample_idx in range(args.sample_times_per_batch):
                att_loss = 0.
                rep_loss = 0.
                rand_seed = int(global_step * args.world_size + sample_idx)  # + args.rank % args.world_size)

                if not args.mlm_loss:
                    if not args.further_train:
                        submodel_config = sample_arch_4_kd(layer_numbers, hidden_sizes, ffn_sizes, qkv_sizes,
                                                           reset_rand_seed=True, rand_seed=rand_seed)
                    # knowledge distillation
                    student_last_rep, student_last_att = student_model(input_ids, submodel_config,
                                                                       attention_mask=input_masks)
                    student_last_att = torch.where(student_last_att <= -1e2,
                                                   torch.zeros_like(student_last_att).to(device),
                                                   student_last_att)

                    att_loss += loss_mse(student_last_att, teacher_last_att)
                    rep_loss += loss_mse(student_last_rep, teacher_last_rep)
                    loss = att_loss + rep_loss

                    if args.gradient_accumulation_steps > 1:
                        rep_loss = rep_loss / args.gradient_accumulation_steps
                        att_loss = att_loss / args.gradient_accumulation_steps
                        loss = loss / args.gradient_accumulation_steps

                    tr_rep_loss += rep_loss.item()
                    tr_att_loss += att_loss.item()
                else:
                    if not args.further_train:
                        submodel_config = sample_arch_4_mlm(layer_numbers, hidden_sizes, ffn_sizes, head_numbers,
                                                            reset_rand_seed=True, rand_seed=rand_seed)
                    loss = student_model(input_ids, submodel_config, attention_mask=input_masks,
                                         masked_lm_labels=lm_label_ids)

                tr_loss += loss.item()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward(retain_graph=True)
                else:
                    loss.backward(retain_graph=True)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if (step + 1) % (args.gradient_accumulation_steps * args.logging_steps) == 0 \
                        and args.local_rank < 2 or global_step < 100:
                    end_time = time.time()

                    if not args.mlm_loss:
                        logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                                'rep_loss is %s; att_loss is %s; (%.2f sec)' %
                                (epoch, global_step + 1, step_in_each_epoch, optimizer.get_lr()[0],
                                 loss.item() * args.gradient_accumulation_steps,
                                 rep_loss.item() * args.gradient_accumulation_steps,
                                 att_loss.item() * args.gradient_accumulation_steps,
                                 end_time - start_time))
                    else:
                        logger.info(
                                'Epoch: %s, global_step: %s/%s, lr: %s, loss is %s; '
                                ' (%.2f sec)' %
                                (epoch, global_step + 1, step_in_each_epoch, optimizer.get_lr()[0],
                                 loss.item() * args.gradient_accumulation_steps,
                                 end_time - start_time))
                    start_time = time.time()

                if args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.local_rank == 0:
                    tb_writer.add_scalar("lr", optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)

                    if not args.mlm_loss:
                        tb_writer.add_scalar("rep_loss", (tr_rep_loss - rep_logging_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("att_loss", (tr_att_loss - att_logging_loss) / args.logging_steps, global_step)
                        rep_logging_loss = tr_rep_loss
                        att_logging_loss = tr_att_loss

                    logging_loss = tr_loss

        # Save a trained model
        if args.rank == 0:
            saving_path = bash_save_dir
            saving_path = Path(os.path.join(saving_path, "epoch_" + str(epoch)))

            if saving_path.is_dir() and list(saving_path.iterdir()):
                logging.warning(f"Output directory ({ saving_path }) already exists and is not empty!")
            saving_path.mkdir(parents=True, exist_ok=True)

            logging.info("** ** * Saving fine-tuned model ** ** * ")
            model_to_save = student_model.module if hasattr(student_model, 'module')\
                else student_model  # Only save the model it-self

            output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
            output_config_file = os.path.join(saving_path, CONFIG_NAME)

            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            args.tokenizer.save_vocabulary(saving_path)

            torch.save(optimizer.state_dict(), os.path.join(saving_path, "optimizer.pt"))
            logger.info("Saving optimizer and scheduler states to %s", saving_path)

            # debug
            if oncloud:
                local_output_dir = os.path.join(LOCAL_DIR, 'output')
                logger.info(mox.file.list_directory(local_output_dir, recursive=True))
                logger.info('s3_output_dir: ' + args.s3_output_dir)
                mox.file.copy_parallel(local_output_dir, args.s3_output_dir)

    if args.local_rank == 0:
        tb_writer.close()


if __name__ == '__main__':
    main()
