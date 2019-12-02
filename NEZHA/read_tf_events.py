###############################
# -*- coding: utf-8 -*-
# Created on Fri Jul 12 11:05:22 2019
# read evaluation results form tf events files and do predict for finetuning
###############################
import os
import tensorflow as tf
import numpy as np
import argparse

acc_list = []
acc_dict = {}
parser = argparse.ArgumentParser()

parser.add_argument('--task_name', help='task_name')
parser.add_argument('--task_data_dir', help='task_data_dir')
parser.add_argument('--max_seq_length', help='max_seq_length')
parser.add_argument('--predict_batch_size', help='predict_batch_size')
parser.add_argument('--pretrained_model_dir', help='pretrained_model_dir')
parser.add_argument('--task_output_dir', help='task_output_dir')
args = parser.parse_args()

# get events files dir
events_name_list = os.listdir(os.path.join(args.task_output_dir, "eval"))
for e in tf.train.summary_iterator(os.path.join(args.task_output_dir, "eval", events_name_list[0])):
    for v in e.summary.value:
        if v.tag == 'eval_accuracy' or v.tag == 'eval_f':
            acc_list.append(v.simple_value)
            acc_dict[v.simple_value] = e.step

# save evaluation results of each checkpoint
results_file = os.path.join(args.task_output_dir, "eval", "all_eval_points_results.txt")
with open(results_file, "w") as writer:
    for ele in acc_list:
        writer.write('step ' + str(acc_dict[ele]) + ': ')
        writer.write(str(ele))
        writer.write('\n')
    writer.write('----best_result: ')
    best_re = np.max(np.array(acc_list))
    writer.write(str(best_re))

# do predict
if 'ner' in args.task_name:
    predict_script = "scripts/run_seq_labelling_predict.sh"
    predict_cmd = ["bash", predict_script, 'ner', args.task_data_dir, args.pretrained_model_dir,
                   args.task_output_dir + 'model.ckpt-' + str(acc_dict[best_re]),
                   args.max_seq_length, args.predict_batch_size, args.task_output_dir]

else:
    predict_script = "scripts/run_clf_predict.sh"
    predict_cmd = ["bash", predict_script, args.task_name, args.task_data_dir, args.pretrained_model_dir,
                   args.task_output_dir + 'model.ckpt-' + str(acc_dict[best_re]),
                   args.max_seq_length, args.predict_batch_size, args.task_output_dir]

try:
    os.system(" ".join(predict_cmd))
except:
    print('error ocurred when doing predict.')
