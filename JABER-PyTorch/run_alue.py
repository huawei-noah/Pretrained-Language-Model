# coding=utf-8
# 2021.09.29 - modified the data loading and modeling part for JABER
#              Huawei Technologies Co., Ltd. 
# source: https://github.com/Alue-Benchmark/alue_baselines/blob/master/bert-baselines/run_alue.py


# Modified version of run_glue script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/examples/run_glue.py

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
""" Finetuning the library models for sequence classification on ALUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import json
import logging
import os
import random
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from NEZHA_PyTorch.modeling_nezha import (
    BertConfig,
    BertForSequenceClassification,
    BertForMultiLabelSequenceClassification 
)

from NEZHA_PyTorch.optimization import BertAdam

from NEZHA_PyTorch.tools import utils
from compute_metrics import alue_compute_metrics as compute_metrics
from processors import alue_convert_examples_to_features as convert_examples_to_features
from processors import alue_output_modes as output_modes
from processors import alue_tasks_label_list as tasks_label_list

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


def set_seed(args):
    if args.seed == -1:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


BEST_ACCURACY = -1
best_pred_dict = {}

def train(args, data, model):
    global BEST_ACCURACY
    global best_pred_dict
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    train_dataset = load_and_cache_examples(args, data, args.task_name, evaluate=False)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps 
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, 
                         lr=args.learning_rate, 
                         warmup=args.warmup_portion,
                         t_total=t_total)

    # Check if saved optimizer states exist
    if os.path.isfile(os.path.join(args.model_path, "optimizer.pt")):
        # Load in optimizer states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for epoch_index in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(epoch_iterator) <= args.gradient_accumulation_steps
                and (step + 1) == len(epoch_iterator)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
                global_step += 1

                # at the end of each epoch
                if args.local_rank in [-1, 0] and (step + 1) == len(epoch_iterator):
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        # if args.task_name != "mq2q":
                        results = evaluate(args, data, model)
                        if args.task_name == 'svreg':
                            metrics = results['pearson']
                        elif args.task_name == 'sec':
                            metrics = results['jaccard']
                        elif args.task_name == 'xnli':
                            metrics = results['acc']
                        else:
                            metrics = results['f1']
                        
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = optimizer.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                    ## save best model
                    output_dir = args.output_dir
                    print("Intermediate evaluate: ", metrics)
                    if BEST_ACCURACY < metrics:
                        BEST_ACCURACY = metrics

                        ### do prediction if do_eval
                        if args.do_eval and args.local_rank in [-1, 0]:
                            evaluate (args,data,model,test=True)

                        ## save the current best ckpt to BestModel.bin 
                        if args.save_model:                
                            checkpoint_name = "pytorch_model.bin"
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training
                            output_model_file = os.path.join(output_dir, checkpoint_name)
                            logger.info("Saving model checkpoint to %s", output_model_file)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            print("Best Model Saved!")
                            output_config_file = os.path.join(output_dir, "bert_config.json")
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())
                        
                            logger.info("Saving optimizer states to %s", output_dir)
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                        # save best eval result
                        output_eval_file = os.path.join(output_dir, "best_eval_results.txt")
                        with open(output_eval_file, "w") as writer:
                            logger.info("***** Best Eval results *****")
                            for key in sorted(results.keys()):
                                logger.info("  %s = %s", key, str(results[key]))
                                writer.write("%s = %s\n" % (key, str(results[key])))
                        
                    
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, data, model, prefix="", test=False):
    # Loop to handle XNLI double evaluation (XNLI-test, XNLI-diag)
    eval_task_names = ("xnli", "xnli-diag") if args.task_name == "xnli" and test==True else (args.task_name,)  # only create (xnli, xnli-diag) for test 
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-diag") if args.task_name == "xnli" else (args.output_dir,)
    eval_mode = "dev" if not test else "test"

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs): 
        eval_dataset = load_and_cache_examples(args, data, eval_task, evaluate=True, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info(f"***** Running {eval_mode} {prefix} *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                inputs["token_type_ids"] = batch[2]

                if len(inputs["labels"][0].shape) == 0 and inputs["labels"][0] == -1:
                    inputs["labels"] = None

                outputs = model(**inputs)
                if test:
                    logits = outputs[0]
                else:
                    tmp_eval_loss, logits = outputs[:2]
                    eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                if not test:
                    out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                if not test:
                    out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
                    

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "multilabel":
            preds = preds > args.threshold
            
        if test:
            # save current prediction into best_pred_dict
            key = "diag" if eval_task == "xnli-diag" else "test"
            best_pred_dict[key] = preds
            
        else:
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def load_and_cache_examples(args, data, task, evaluate=False, test=False):
    '''
    ALUE:
    - dev
        - MDD: read "test"
    - test
        -XNLI: both "test" and "diag"
    '''
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    main_task = task.split("-")[0] 
    output_mode = output_modes[main_task]
    # Load data features from cache or dataset file
    data_type = "dev" if evaluate else "train"
    if test: data_type = "test"
    cached_features_file = os.path.join(
        "features",
        "cached_{}_{}_{}_{}".format(
            data_type,
            list(filter(None, args.model_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if not os.path.exists("features"):
        os.makedirs("features")
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        label_list = tasks_label_list[main_task]
        data_key = main_task.upper() 
        if test:
            if task == "xnli-diag":
                examples = data[data_key]['diag']
            else:
                examples = data[data_key]['test'] 
        elif evaluate:
            examples = data[data_key]['dev'] 
        else:           
            examples = data[data_key]['train']  

        features = convert_examples_to_features(
            examples,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,  # pad on the left for xlnet
            pad_token=0,
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  
    if test:
        all_labels = torch.tensor([-1 for f in features], dtype=torch.long)
    elif output_mode in ["classification", "multilabel"]:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    
        
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="path to the fine-tuning .pkl file",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="name of the model (e.g. pytorch_model-835820.bin",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_name",
        default="alue.log",
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        default=0.1,
        type=float,
        help="The hidden_dropout_prob in bert config file",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--threshold",
        default=0.25,
        type=float,
        help="The threshold used for interpreting probabilities to class labels in multilabel tasks",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_portion", default=0, type=float, help="Linear warmup over total_steps*warmup_portion.")

    parser.add_argument("--logging_steps", type=int, default=-1, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_model", type=bool, default=True, help="Wheter to save the current best model or not.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare ALUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in tasks_label_list:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]
    label_list = tasks_label_list[args.task_name]
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    
    print('init model...')
    ## rewrite the config file (hidden_dropout_prob)
    config_path = os.path.join(args.model_path,'bert_config.json')
    with open(config_path, "r") as jsonFile:
        bert_config = json.load(jsonFile)

    bert_config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config_path = os.path.join(args.output_dir, "config_bert_%s_%s.json" % (args.model_name, args.task_name))
    with open(config_path, "w") as jsonFile:
        json.dump(bert_config, jsonFile)
    
    config = BertConfig.from_json_file(config_path)
    os.remove(config_path)
    
    if args.output_mode == 'multilabel':
        pretrained_model_prediction_type = BertForMultiLabelSequenceClassification
    else:
        pretrained_model_prediction_type = BertForSequenceClassification

    model = pretrained_model_prediction_type(config,num_labels=num_labels)
    utils.torch_show_all_params(model)
    utils.torch_init_model(model, os.path.join(args.model_path, args.model_name))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    ### load data
    data = pickle.load(open(args.data, "rb"))

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, data, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    # save the best results for each task
    fout = open("./%s" % args.log_name, 'a')
    ckpt = args.model_name.split("-")[-1].split(".")[0]
    hp_list = ["per_gpu_train_batch_size", "learning_rate", "hidden_dropout_prob"]
    exp_name = ckpt + " " + " ".join(["%s %s" % (hp, getattr(args, hp)) for hp in hp_list])
    fout.write("%s\t%s\t%s\t%.2f\n" % ("JABER", args.task_name.upper(), exp_name, BEST_ACCURACY*100))
    fout.close()

    # save prediction for the best checkpoint
    if args.do_eval and args.local_rank in [-1, 0]:
        data_key = args.task_name.upper()
        for portion, y_pred in best_pred_dict.items():
            idx_lst = [exp["idx"] for exp in data[data_key][portion]]
            best_pred_dict[portion] = [(x, y) for x, y in zip(idx_lst, y_pred)]
        key = ("JABER", args.task_name.upper(), "%.2f" % (BEST_ACCURACY*100))
        pred_path = "./alue_predictions"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        filename = os.path.join(pred_path, "%s_%s_%s.pkl" % key)
        with open(filename, 'wb') as handle:
            pickle.dump(best_pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
