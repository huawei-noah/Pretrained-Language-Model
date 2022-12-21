#!/bin/bash

export CUDA_VISIBLE_DEVICES="7"
model_name="JABER"
task_name="OOLD"
per_gpu_train_batch_size="128"
dropout_rate="0.2"
learning_rate="2e-05"
#task_name="OHSD"
#per_gpu_train_batch_size="32"
#dropout_rate="0.3"
#learning_rate="7e-06"
#model_name = $1
#task_name=$2
#per_gpu_train_batch_size=$3
#dropout_rate=$4
#learning_rate=$5
output_dir="/tmp/${model_name}_${task_name}"
log_file="/tmp/test.log"
model_path="./pretrained_models/${model_name}/"
PYTHONIOENCODING=UTF-8 python run_alue.py \
    --model_name $model_name \
    --step -1 \
    --task_name $task_name \
    --output_dir $output_dir \
    --do_train \
    --learning_rate $learning_rate \
    --dropout_rate $dropout_rate \
    --weight_decay 0.01 \
    --num_train_epochs 30 \
    --warmup_portion 0.1 \
    --save_epochs True \
    --seed -1 \
    --evaluate_during_training \
    --per_gpu_train_batch_size $per_gpu_train_batch_size \
    --per_gpu_eval_batch_size $per_gpu_train_batch_size \
    --gradient_accumulation_steps 1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --log_file $log_file \
    --model_path $model_path \
    --save_model 0