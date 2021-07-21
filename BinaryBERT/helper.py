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
import logging
import os
import string
import random
import torch

def generate_job_id():
  return ''.join(random.sample(string.ascii_letters+string.digits, 5))

def init_logging(log_path):

  if not os.path.isdir(os.path.dirname(log_path)):
    print("Log path does not exist. Create a new one.")
    os.makedirs(os.path.dirname(log_path))
  if os.path.exists(log_path):
    print("%s already exists. replace it with current experiment." % log_path)
    os.system('rm %s' % log_path)

  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

  fileHandler = logging.FileHandler(log_path)
  fileHandler.setFormatter(logFormatter)
  logger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  logger.addHandler(consoleHandler)

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        logging.info("{0}: {1}".format(k, v))

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()


def visualize_clip(clip_dict):
    # assert len(clip_dict) > 0, 'empty clip_dict, possibly not learnable_scalling.'
    logging.info("Visualizing learnable clipping vals...")
    for n, p in clip_dict.items():
        if p.nelement() == 2:
            # PACT clip val has two elements
            logging.info("PACT clip_val: %s: (%.4f, %.4f)" % (n, p[0].item(), p[1].item()))
        elif p.nelement() == 1:
            # LSQ step size has only one element
            logging.info("LSQ step_size: %s: %.4f" % (n, p.item()))


def result_to_file(result, file_name):
    with open(file_name, "a") as writer:
        logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            if result[key]>0.0:
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
