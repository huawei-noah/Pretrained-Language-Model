# Modified version of run_glue script
# Source: https://github.com/huggingface/transformers/blob/v2.7.0/examples/run_glue.py

# coding=utf-8
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
""" Finetuning the library models for sequence classification on ALUE (T5)."""


import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from transformers import (
    AdamW,
    AutoConfig,
    T5ForConditionalGeneration,
    get_constant_schedule,
    Adafactor,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from generate_data import *

logger = logging.getLogger(__name__)


class MyDataCollator:
    def __init__(self, args, data_processor: DataProcessor):
        self.task_type = data_processor.task_type
        self.is_gen = data_processor.is_gen
        self.arch = args.arch

        self.pad_idx = data_processor.pad_idx
        self.bos_idx, self.eos_idx = data_processor.bos_idx, data_processor.eos_idx

    def __call__(self, features):
        batch = self._pad([f["input_ids"] for f in features])

        if self.task_type in ["mlc", "gen"] or self.is_gen:
            labels = [f["labels"] for f in features]
        else:
            labels = [f["labels"][0] for f in features]

        if self.is_gen:
            batch.update(self._pad_labels(labels))
        else:
            dtype = torch.float if self.task_type == "reg" and not self.is_gen else torch.long
            batch["labels"] = torch.tensor(labels, dtype=dtype)

        return batch

    def _pad(self, input_ids_lst):
        msq = max(len(lst) for lst in input_ids_lst)
        input_ids, attention_mask = [], []
        for ii in input_ids_lst:
            val = msq - len(ii)
            input_ids.append(ii + [self.pad_idx] * val)
            attention_mask.append([1] * len(ii) + [0] * val)
        input_ids = torch.as_tensor(input_ids, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.arch == "bert":
            batch["token_type_ids"] = self.get_segment_ids(input_ids)

        return batch

    def _pad_labels(self, labels_lst):

        msq = max(len(lst) for lst in labels_lst)
        labels, decoder_attention_mask, decoder_input_ids = [], [], []
        for lbl in labels_lst:
            seq_len = len(lbl)
            labels.append(lbl + [self.eos_idx] + [-100] * (msq - seq_len))
            decoder_input_ids.append([self.pad_idx]+lbl + [0] * (msq - seq_len)) #self.bos_idx
            decoder_attention_mask.append([1] * (seq_len+1) + [0] * (msq - seq_len))

        labels = torch.as_tensor(labels, dtype=torch.long)
        decoder_attention_mask = torch.as_tensor(decoder_attention_mask, dtype=torch.long)
        decoder_input_ids = torch.as_tensor(decoder_input_ids, dtype=torch.long)
        return {"labels": labels,
                "decoder_input_ids": decoder_input_ids, "decoder_attention_mask": decoder_attention_mask}

    def get_segment_ids(self, input_ids):
        batch_size, max_seq_len = input_ids.shape
        rng = torch.tile(torch.unsqueeze(torch.arange(max_seq_len), 0),
                         (batch_size, 1))  # , device=torch.device('cuda')
        zero_arr, one_arr = torch.zeros_like(input_ids), torch.ones_like(input_ids)
        cond = input_ids == (self.eos_idx * one_arr)
        segment_ids, _ = torch.min(torch.where(cond, rng, max_seq_len * one_arr), dim=-1)
        segment_ids = torch.tile(torch.unsqueeze(segment_ids, 1), (1, max_seq_len))
        cond = torch.logical_and(torch.greater(rng, segment_ids), torch.not_equal(input_ids, zero_arr))
        segment_ids = torch.where(cond, one_arr, zero_arr)
        return segment_ids


def set_seed(args):
    if args.seed == -1:
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

BEST_ACCURACY = -1
BEST_PRED_DICT = {}

def train(args, data_processor: DataProcessor, model, data_collator: MyDataCollator):
    global BEST_ACCURACY
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    dataset_dict, sampler_dict, dataloader_dict = load_dataloaders(args, data_processor, data_collator)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(dataloader_dict["train"]) // args.gradient_accumulation_steps) + 1
    else:
        args.logging_steps = len(dataloader_dict["train"]) // args.gradient_accumulation_steps
        t_total = len(dataloader_dict["train"]) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "layer_norm.weight"]#
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    if args.arch == "bert":
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    else:
        optimizer = Adafactor(model.parameters(),  # optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              relative_step=False,
                              warmup_init=False)
    scheduler = get_constant_schedule(optimizer)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_path, "scheduler.pt")))

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
    logger.info("  Num examples = %d", len(dataset_dict["train"]))
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
        epochs_trained = global_step // (len(dataloader_dict["train"]) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(dataloader_dict["train"]) // args.gradient_accumulation_steps)

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
        epoch_iterator = tqdm(dataloader_dict["train"], desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {k: v.to(args.device) for k, v in batch.items()}
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
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step == 1:
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                if args.local_rank in [-1, 0] and args.save_epochs and (step + 1) == len(epoch_iterator):# or True
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well

                        eval_loss, results, metrics = \
                            run_eval(args, data_processor, model, dataset_dict, dataloader_dict)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                        data_processor.reset_pred()
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
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
                        global BEST_PRED_DICT

                        if args.save_model:
                            checkpoint_name = "pytorch_model.bin"

                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )  # Take care of distributed/parallel training

                            output_model_file = os.path.join(args.output_dir, checkpoint_name)
                            logger.info("Saving model checkpoint to %s", output_model_file)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            print("Best Model Saved!")
                            output_config_file = os.path.join(args.output_dir, "config.json")
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())

                            # logger.info("Saving optimizer states to %s", output_dir)
                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))

                        # save best result
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


def run_eval(args, data_processor: DataProcessor, model, dataset_dict, dataloader_dict):
    # eval_dataloader, eval_sample_num, portion
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    eval_loss = 0.0

    portion_lst = ["dev"] + [p for p in dataloader_dict.keys() if p not in ["dev", "train"]]
    dev_results, dev_metrics = None, -1

    for portion in portion_lst:
        if portion != "dev" and dev_metrics < BEST_ACCURACY: continue
        logger.info("***** Running evaluation {} *****".format(portion))
        logger.info("  Num examples = %d", len(dataset_dict[portion]))
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_eval_steps = 0

        for batch in tqdm(dataloader_dict[portion], desc="Evaluating"):
            model.eval()
            inputs = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                if data_processor.is_gen:
                    decoder_seq_len = inputs["labels"].shape[-1]
                    inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
                    obj = model.module if args.n_gpu > 1 else model
                    decoder_output_ids = obj.generate(**inputs,
                                                               num_beams=5,
                                                               max_length=decoder_seq_len,
                                                               early_stopping=True
                                                        )
                    decoder_output_ids = decoder_output_ids.detach().cpu().numpy().tolist()
                    data_processor.y_pred[portion].append(decoder_output_ids)

                else:
                    model_output = model(**inputs)
                    tmp_eval_loss, logits = model_output[:2]
                    eval_loss += tmp_eval_loss.mean().item()
                    logits = logits.detach().cpu().numpy().tolist()
                    data_processor.y_logits[portion].append(logits)

            nb_eval_steps += 1
        data_processor.process_logits(portion)
        if portion == "dev":
            dev_results = data_processor.compute_score(portion)
            dev_metrics = data_processor.final_metric(dev_results)
            eval_loss = eval_loss / nb_eval_steps
            logger.info("***** Eval results {} *****".format(portion))
            logger.info("  %s = %s", "eval_loss", str(eval_loss))

            for key in sorted(dev_results.keys()):
                logger.info("  %s = %s", key, str(dev_results[key]))
        else:
            key = "diag" if args.task_name == "xnli-diag" else "test"
            BEST_PRED_DICT[key] = data_processor.y_pred[portion]
            if data_processor.y_true[portion]:
                results = data_processor.compute_score(portion)
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))

    return eval_loss, dev_results, dev_metrics


def load_model(args, data_processor: DataProcessor):
    ckpt_name = "pytorch_model-%s.bin" % args.step if args.step != "-1" else "pytorch_model.bin"

    def get_disc_lbl_num():
        if data_processor.task_type == "mlc": return len(MLC_LBL_DICT[data_processor.task_name])
        elif data_processor.task_type == "ner": return len(data_processor.id2label)
        return len(data_processor.id2label) if data_processor.id2label else 1

    if args.arch == "bert":
        num_labels = get_disc_lbl_num()
        model = _load_jaber(args, data_processor.task_type, num_labels)

    elif args.arch == "t5":
        t5_config = AutoConfig.from_pretrained(args.model_path, dropout_rate=args.dropout_rate)
        if data_processor.is_gen:
            model = T5ForConditionalGeneration.from_pretrained(
                os.path.join(args.model_path, ckpt_name),
                config=t5_config)

    else:
        raise ValueError("Unsupported Model arch `%s`" % args.arch)

    return model


def _load_jaber(args, task_type, num_labels):
    from NEZHA_PyTorch.modeling_nezha import (
        BertConfig,
        BertForSequenceClassification,
        BertForMultiLabelSequenceClassification
    )
    from NEZHA_PyTorch.tools import utils

    print('init model...')
    ## rewrite the config file (hidden_dropout_prob)
    config_path = os.path.join(args.model_path, 'bert_config.json')
    with open(config_path, "r") as jsonFile:
        bert_config = json.load(jsonFile)

    bert_config["hidden_dropout_prob"] = args.dropout_rate
    config_path = os.path.join("/tmp/", "config_bert_%s_%s.json" % (args.model_name, args.task_name))
    with open(config_path, "w") as jsonFile:
        json.dump(bert_config, jsonFile)

    config = BertConfig.from_json_file(config_path)
    os.remove(config_path)

    if task_type == "mlc":
        pretrained_model_prediction_type = BertForMultiLabelSequenceClassification
    else:
        pretrained_model_prediction_type = BertForSequenceClassification
    model = pretrained_model_prediction_type(config, num_labels=num_labels)

    utils.torch_show_all_params(model)
    utils.torch_init_model(model, os.path.join(args.model_path, "pytorch_model.bin"))

    return model


def load_initials(args):
    # load the dp class
    key = (args.task_name, args.model_name)
    filename = os.path.join("./raw_datasets", "dp.%s.%s.pkl" % key)
    with open(filename, 'rb') as fp:
        data_processor = pickle.load(fp)

    # load model
    model = load_model(args, data_processor)
    model.to(args.device)

    # init data collator
    data_collator = MyDataCollator(args, data_processor)

    return data_processor, model, data_collator


def load_dataloaders(args, data_processor, data_collator, overwrite_train=False):
    dataset_dict, sampler_dict, dataloader_dict = {}, {}, {}

    # load train dataset and dataloader
    portion = "train"
    key = ("dataset", data_processor.task_name, data_processor.model_name, data_processor.is_gen, portion)
    cache_url = os.path.join("./raw_datasets", "%s_%s_%s_%s_%s" % key)
    dataset_dict[portion] = datasets.load_from_disk(cache_url, keep_in_memory=False)
    if args.local_rank == -1:
        sampler_dict[portion] = RandomSampler(dataset_dict[portion])
    else:
        sampler_dict[portion] = DistributedSampler(dataset_dict[portion])
    dataloader_dict[portion] = DataLoader(
        dataset_dict[portion], sampler=sampler_dict[portion],
        collate_fn=data_collator, batch_size=args.per_gpu_train_batch_size
    )

    # load dataset and dataloader for no train portion
    for portion in data_processor.data_dict.keys():
        if not overwrite_train and portion == "train": continue
        key = ("dataset", data_processor.task_name, data_processor.model_name, data_processor.is_gen, portion)
        cache_url = os.path.join("./raw_datasets", "%s_%s_%s_%s_%s" % key)
        dataset_dict[portion] = datasets.load_from_disk(cache_url, keep_in_memory=False)
        sampler_dict[portion] = SequentialSampler(dataset_dict[portion])
        dataloader_dict[portion] = DataLoader(
            dataset_dict[portion], sampler=sampler_dict[portion], collate_fn=data_collator,
            batch_size=args.per_gpu_eval_batch_size
        )

    return dataset_dict, sampler_dict, dataloader_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dropout_rate",
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
    parser.add_argument("--save_epochs", type=bool, default=True, help="Save checkpoint at the end of each epoch.")
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

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="name of the model (e.g. pytorch_model-835820.bin)",
    )
    parser.add_argument(
        "--task_name",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_file",
        default=None,
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "--step",
        default=None,
        type=str,
        help="The name of the log file",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--save_model",
        default=0,
        type=int,
        help="save bets ckpt or not",
    )

    args = parser.parse_args()

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

    for arch, model_name_set in MODEL_ARCH_MAP.items():
        if args.model_name in model_name_set:
            args.arch = arch
            break

    # if args.arch == "t5":
    #     args.per_gpu_eval_batch_size = 8
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    data_processor, model, data_collator = load_initials(args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, data_processor, model, data_collator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # save the best results for each model_type
    fout = open(args.log_file, 'a')
    hp_str = " ".join(["%s:%s" % (hp, getattr(args, hp)) for hp in HP_LST if hasattr(args, hp)])
    fout.write("%s\t%s\t%s\t%.2f\t%s\n" % (args.model_name, args.step, args.task_name, BEST_ACCURACY, hp_str))
    fout.close()

    # save prediction for the best checkpoint
    score_ext = "%.2f" % BEST_ACCURACY
    if args.task_name in ALUE_TASKS and args.local_rank in [-1, 0]:
        pred_path = "./alue_predictions/%s" % args.model_name
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        for portion, y_pred in BEST_PRED_DICT.items():
            raw_dataset_dir = "./raw_datasets"
            save_alue_leaderboard(raw_dataset_dir, pred_path, args.task_name, portion, y_pred, score_ext)



if __name__ == "__main__":
    main()
