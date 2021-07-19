# coding=utf-8
# Copyright 2021 Huawei Technologies Co., Ltd.
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
import torch
import logging
from transformer.file_utils import WEIGHTS_NAME, CONFIG_NAME, MISC_NAME
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from transformer.optimization import BertAdam
from helper import *
from utils_glue import *
import numpy as np
import pickle
logging.basicConfig(level=logging.INFO)


class KDLearner(object):
    def __init__(self, args, device, student_model, teacher_model=None,num_train_optimization_steps=None):
        self.args = args
        self.device = device
        self.n_gpu = torch.cuda.device_count()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_train_optimization_steps = num_train_optimization_steps
        self._check_params()

    def build(self, lr=None):
        self.prev_global_step = 0
        if self.args.distill_rep_attn and not self.args.distill_logit:
            stage = 'kd_stage1'
        elif self.args.distill_logit and not self.args.distill_rep_attn:
            stage = 'kd_stage2'
        elif self.args.distill_logit and self.args.distill_rep_attn:
            stage = 'kd_joint'
        else:
            stage = 'nokd'
        self.output_dir = os.path.join(self.args.output_dir, stage)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        param_optimizer = list(self.student_model.named_parameters())
        self.clip_params = {}
        for k, v in param_optimizer:
            if 'clip_' in k:
                self.clip_params[k] = v

        # if self.args.input_quant_method == 'uniform' and self.args.restore_clip:
        #     self._restore_clip_params()
        # elif self.args.input_quant_method == 'uniform':
        #     logging.info("All clipping vals initialized at (%.4f, %.4f)" % (-self.args.clip_init_val, self.args.clip_init_val))
        # else:
        #     pass

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and not 'clip_' in n)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and not 'clip_' in n)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.clip_params.items()], 'lr': self.args.clip_lr, 'weight_decay': self.args.clip_wd},
        ]

        schedule = 'warmup_linear'
        learning_rate = self.args.learning_rate if not lr else lr
        self.optimizer = BertAdam(optimizer_grouped_parameters,
                             schedule=schedule,
                             lr=learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=self.num_train_optimization_steps)
        logging.info("Optimizer prepared.")
        self._check_quantized_modules()
        self._setup_grad_scale_stats()

    def _do_eval(self, model, task_name, eval_dataloader, output_mode, eval_labels, num_labels):
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for batch_ in tqdm(eval_dataloader, desc="Evaluating"):
            batch_ = tuple(t.to(self.device) for t in batch_)
            with torch.no_grad():
                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

                logits, _, _ = model(input_ids, segment_ids, input_mask)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, eval_labels.numpy())
        result['eval_loss'] = eval_loss

        return result

    def evaluate(self, task_name, eval_dataloader, output_mode, eval_labels, num_labels, eval_examples,
                 mm_eval_dataloader, mm_eval_labels):
        """ Evalutaion of checkpoints from models/. directly use args.student_model """

        self.student_model.eval()
        result = self._do_eval(self.student_model, task_name, eval_dataloader, output_mode, eval_labels, num_labels)

        logging.info("***** Running evaluation, Task: %s, Job_id: %s *****" % (self.args.task_name, self.args.job_id))
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", self.args.batch_size)
        logging.info("***** Eval results, Task: %s, Job_id: %s *****" % (self.args.task_name, self.args.job_id))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))

        if task_name == "mnli":
            logging.info('MNLI-mm Evaluation')
            result = self._do_eval(self.student_model, 'mnli-mm', mm_eval_dataloader, output_mode, mm_eval_labels, num_labels)
            tmp_output_eval_file = os.path.join(self.args.output_dir + '-MM', "eval_results.txt")
            result_to_file(result, tmp_output_eval_file)

    def train(self, train_examples, task_name, output_mode, eval_labels, num_labels,
                    train_dataloader, eval_dataloader, eval_examples, tokenizer, mm_eval_labels, mm_eval_dataloader):
        """ quant-aware pretraining + KD """

        # Prepare loss functions
        loss_mse = MSELoss()

        self.teacher_model.eval()
        teacher_results = self._do_eval(self.teacher_model, task_name, eval_dataloader, output_mode, eval_labels, num_labels)
        logging.info("Teacher network evaluation")
        for key in sorted(teacher_results.keys()):
            logging.info("  %s = %s", key, str(teacher_results[key]))

        self.teacher_model.train()  # switch to train mode to supervise students

        # Train and evaluate
        # num_layers = self.student_model.config.num_hidden_layers + 1
        global_step = self.prev_global_step
        best_dev_acc = 0.0
        output_eval_file = os.path.join(self.args.output_dir, "eval_results.txt")

        logging.info("***** Running training, Task: %s, Job id: %s*****" % (self.args.task_name, self.args.job_id))
        logging.info(" Distill rep attn: %d, Distill logit: %d" % (self.args.distill_rep_attn, self.args.distill_logit))
        logging.info("  Num examples = %d", len(train_examples))
        logging.info("  Batch size = %d", self.args.batch_size)
        logging.info("  Num steps = %d", self.num_train_optimization_steps)

        global_tr_loss = 0  # record global average training loss to plot

        for epoch_ in range(self.args.num_train_epochs):

            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(train_dataloader):

                self.student_model.train()

                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                rep_loss_layerwise = []
                att_loss_layerwise = []

                student_logits, student_atts, student_reps = self.student_model(input_ids, segment_ids, input_mask)

                if self.args.distill_logit or self.args.distill_rep_attn:
                    # use distillation

                    with torch.no_grad():
                        teacher_logits, teacher_atts, teacher_reps = self.teacher_model(input_ids, segment_ids, input_mask)

                    # NOTE: config loss according to stage
                    loss = 0.
                    if self.args.distill_logit:
                        cls_loss = soft_cross_entropy(student_logits / self.args.temperature,
                                                      teacher_logits / self.args.temperature)
                        loss += cls_loss
                        tr_cls_loss += cls_loss.item()

                    if self.args.distill_rep_attn:
                        for student_att, teacher_att in zip(student_atts, teacher_atts):
                            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(self.device),
                                                      student_att)
                            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(self.device),
                                                      teacher_att)

                            tmp_loss = loss_mse(student_att, teacher_att)
                            att_loss += tmp_loss
                            att_loss_layerwise.append(tmp_loss.item())

                        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                            tmp_loss = loss_mse(student_rep, teacher_rep)
                            rep_loss += tmp_loss
                            rep_loss_layerwise.append(tmp_loss.item())

                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()

                        loss += rep_loss + att_loss

                else:
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(student_logits, label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_mse = MSELoss()
                        loss = loss_mse(student_logits.view(-1), label_ids.view(-1))

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                global_tr_loss += loss.item()
                nb_tr_examples += label_ids.size(0)
                nb_tr_steps += 1

                # evaluation and save model
                if global_step % self.args.eval_step == 0 or \
                        global_step == len(train_dataloader)-1:

                    

                    # logging.info("***** KDLearner %s Running evaluation, Task: %s, Job_id: %s *****" % (stage, self.args.task_name, self.args.job_id))
                    logging.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logging.info("  Num examples = %d", len(eval_examples))
                    logging.info(f"  Previous best = {best_dev_acc}")

                    loss = tr_loss / (step + 1)
                    global_avg_loss = global_tr_loss / (global_step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    self.student_model.eval()
                    result = self._do_eval(self.student_model, task_name, eval_dataloader, output_mode, eval_labels, num_labels)
                    result['global_step'] = global_step
                    result['cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss
                    result['global_loss'] = global_avg_loss

                    preds = student_logits.detach().cpu().numpy()
                    train_label = label_ids.cpu().numpy()
                    if output_mode == "classification":
                        preds = np.argmax(preds, axis=1)
                    elif output_mode == "regression":
                        preds = np.squeeze(preds)
                    result['train_batch_acc'] = list(compute_metrics(task_name, preds, train_label).values())[0]

                    if self.args.distill_rep_attn:
                        logging.info("embedding layer rep_loss: %.8f" % (rep_loss_layerwise[0]))
                        rep_loss_layerwise = rep_loss_layerwise[1:]
                        for lid in range(len(rep_loss_layerwise)):
                            logging.info("layer %d rep_loss: %.8f" % (lid+1, rep_loss_layerwise[lid]))
                            logging.info("layer %d att_loss: %.8f" % (lid+1, att_loss_layerwise[lid]))

                    result_to_file(result, output_eval_file)

                    save_model = False

                    if task_name in acc_tasks and result['acc'] > best_dev_acc:
                        best_dev_acc = result['acc']
                        save_model = True

                    if task_name in corr_tasks and result['corr'] > best_dev_acc:
                        best_dev_acc = result['corr']
                        save_model = True

                    if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                        best_dev_acc = result['mcc']
                        save_model = True

                    if save_model:
                        self._save()

                        if task_name == "mnli":
                            logging.info('MNLI-mm Evaluation')
                            result = self._do_eval(self.student_model, 'mnli-mm', mm_eval_dataloader, output_mode, mm_eval_labels, num_labels)
                            result['global_step'] = global_step
                            tmp_output_eval_file = os.path.join(self.output_dir + '-MM', "eval_results.txt")
                            result_to_file(result, tmp_output_eval_file)

                # if self.args.quantize_weight:
                    # self.quanter.restore()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1

    def _save(self):
        logging.info("******************** Save model ********************")
        model_to_save = self.student_model.module if hasattr(self.student_model, 'module') else self.student_model
        output_model_file = os.path.join(self.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(self.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)

    def _check_params(self):
        if not self.args.do_eval:
            assert self.teacher_model, 'teacher model must not be None in train mode.'

    def _check_quantized_modules(self):
        logging.info("Checking module types.")
        for k, m in self.student_model.named_modules():
            if isinstance(m, torch.nn.Linear):
                logging.info('%s: %s' % (k, str(m)))

    def _setup_grad_scale_stats(self):
        self.grad_scale_stats = {'weight': None, \
                                 'bias': None, \
                                 'layer_norm': None, \
                                 'step_size/clip_val': None}
        self.ema_grad = 0.9

    def check_grad_scale(self):
        logging.info("Check grad scale ratio: grad/w")
        for k, v in self.student_model.named_parameters():
            if v.grad is not None:
                has_grad = True
                ratio = v.grad.norm(p=2) / v.data.norm(p=2)
                # print('%.6e, %s' % (ratio.float(), k))
            else:
                has_grad = False
                logging.info('params: %s has no gradient' % k)
                continue

            # update grad_scale stats
            if 'weight' in k and v.ndimension() == 2:
                key = 'weight'
            elif 'bias' in k and v.ndimension() == 1:
                key = 'bias'
            elif 'LayerNorm' in k and 'weight' in k and v.ndimension() == 1:
                key = 'layer_norm'
            elif 'clip_' in k:
                key = 'step_size/clip_val'
            else:
                key = None

            if key and has_grad:
                if self.grad_scale_stats[key]:
                    self.grad_scale_stats[key] = self.ema_grad * self.grad_scale_stats[key] + (1-self.ema_grad) * ratio
                else:
                    self.grad_scale_stats[key] = ratio

        for (key, val) in self.grad_scale_stats.items():
            if val is not None:
                logging.info('%.6e, %s' % (val, key))
