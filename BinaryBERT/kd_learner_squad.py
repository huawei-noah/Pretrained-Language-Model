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
from utils_squad import *
import numpy as np
import pickle
logging.basicConfig(level=logging.INFO)


class KDLearner(object):
    def __init__(self, args, device, student_model, teacher_model=None, num_train_optimization_steps = None):
        self.args = args
        self.device = device
        self.n_gpu = torch.cuda.device_count()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.num_train_optimization_steps = num_train_optimization_steps
        self._check_params()
        self.name = 'kd_'   # learner suffix for saving

    def build(self, lr=None):
        self.prev_global_step = 0
        if self.args.distill_rep_attn and not self.args.distill_logit:
            self.stage = 'kd_stage1'
        elif self.args.distill_logit and not self.args.distill_rep_attn:
            self.stage = 'kd_stage2'
        elif self.args.distill_logit and self.args.distill_rep_attn:
            self.stage = 'kd_joint'
        else:
            self.stage = 'nokd'
        self.output_dir = os.path.join(self.args.output_dir, self.stage)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        param_optimizer = list(self.student_model.named_parameters())
        self.clip_params = {}
        for k, v in param_optimizer:
            if 'clip_' in k:
                self.clip_params[k] = v

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

    def eval(self, model,dataloader,features,examples,dataset):
        all_results = []
        for _,batch_ in tqdm(enumerate(dataloader)):
            batch_ = tuple(t.to(self.device) for t in batch_)
            input_ids, input_mask, segment_ids, example_indices = batch_
            with torch.no_grad():
                (batch_start_logits, batch_end_logits),_,_ = model(input_ids, segment_ids, input_mask)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                    start_logits=start_logits,
                                    end_logits=end_logits))

        return write_predictions(examples, features, all_results,
                    self.args.n_best_size, self.args.max_answer_length,
                    True, False,
                    self.args.version_2_with_negative, self.args.null_score_diff_threshold,dataset)

    def train(self, train_dataloader, eval_dataloader, eval_features, eval_examples, dev_dataset):
        """ quant-aware pretraining + KD """

        # Prepare loss functions
        loss_mse = MSELoss()
        self.teacher_model.eval()
        teacher_results = self.eval(self.teacher_model, eval_dataloader,eval_features,eval_examples, dev_dataset)
        logging.info("Teacher network evaluation")
        for key in sorted(teacher_results.keys()):
            logging.info("  %s = %s", key, str(teacher_results[key]))

        # self.teacher_model.train()  # switch to train mode to supervise students

        # Train and evaluate
        # num_layers = self.student_model.config.num_hidden_layers + 1
        global_step = 0
        best_dev_f1 = 0.0
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")

        logging.info(" Distill rep attn: %d, Distill logit: %d" % (self.args.distill_rep_attn, self.args.distill_logit))
        logging.info("  Batch size = %d", self.args.batch_size)
        logging.info("  Num steps = %d", self.num_train_optimization_steps)

        global_tr_loss = 0  # record global average training loss to plot

        for epoch_ in range(int(self.args.num_train_epochs)):

            tr_loss = 0.
            tr_att_loss = 0.
            tr_rep_loss = 0.
            tr_cls_loss = 0.

            for step, batch in enumerate(train_dataloader):

                self.student_model.train()
                batch = tuple(t.to(self.device) for t in batch)

                input_ids, input_mask, segment_ids, start_positions, end_positions = batch

                att_loss = 0.
                rep_loss = 0.
                cls_loss = 0.
                rep_loss_layerwise = []
                att_loss_layerwise = []
                loss = 0.
                if self.args.distill_logit or self.args.distill_rep_attn:
                    # use distillation
                    student_logits, student_atts, student_reps = self.student_model(input_ids, segment_ids, input_mask)
                    with torch.no_grad():
                        teacher_logits, teacher_atts, teacher_reps = self.teacher_model(input_ids, segment_ids, input_mask)

                    # NOTE: config loss according to stage
                    if self.args.distill_logit:
                        soft_start_ce_loss = soft_cross_entropy(student_logits[0], teacher_logits[0])
                        soft_end_ce_loss = soft_cross_entropy(student_logits[1], teacher_logits[1])
                        cls_loss = soft_start_ce_loss+soft_end_ce_loss
                        loss+=cls_loss
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
                        # rep_loss_layerwise = rep_loss_layerwise[1:]  # remove embed dist

                        tr_att_loss += att_loss.item()
                        tr_rep_loss += rep_loss.item()
                        loss += rep_loss + att_loss

                else:
                    cls_loss, _, _ = self.student_model(input_ids, segment_ids, input_mask,start_positions, end_positions)
                    loss+=cls_loss
                    tr_cls_loss += cls_loss.item()

                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                global_tr_loss += loss.item()

                # evaluation and save model
                if global_step % self.args.eval_step == 0 or \
                        global_step == len(train_dataloader)-1:

                    logging.info("***** KDLearner %s Running evaluation, Job_id: %s *****" % (self.stage, self.args.job_id))
                    logging.info("  Epoch = {} iter {} step".format(epoch_, global_step))
                    logging.info(f"  Previous best = {best_dev_f1}")

                    loss = tr_loss / (step + 1)
                    global_avg_loss = global_tr_loss / (global_step + 1)
                    cls_loss = tr_cls_loss / (step + 1)
                    att_loss = tr_att_loss / (step + 1)
                    rep_loss = tr_rep_loss / (step + 1)

                    self.student_model.eval()
                    result = self.eval(self.student_model, eval_dataloader,eval_features,eval_examples, dev_dataset)
                    result['global_step'] = global_step
                    result['train_cls_loss'] = cls_loss
                    result['att_loss'] = att_loss
                    result['rep_loss'] = rep_loss
                    result['loss'] = loss
                    result['global_loss'] = global_avg_loss

                    if self.args.distill_rep_attn:
                        # add the layerwise loss on rep and att
                        logging.info("embedding layer rep_loss: %.8f" % (rep_loss_layerwise[0]))
                        rep_loss_layerwise = rep_loss_layerwise[1:]
                        for lid in range(len(rep_loss_layerwise)):
                            logging.info("layer %d rep_loss: %.8f" % (lid+1, rep_loss_layerwise[lid]))
                            logging.info("layer %d att_loss: %.8f" % (lid+1, att_loss_layerwise[lid]))

                    result_to_file(result, output_eval_file)

                    save_model = False

                    if result['f1'] > best_dev_f1:
                        best_dev_f1 = result['f1']
                        save_model = True

                    if save_model:
                        self._save()

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
