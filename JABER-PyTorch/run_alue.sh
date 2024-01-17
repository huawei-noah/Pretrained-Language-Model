#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

model_name="JABER"
task_name="abusive"
per_gpu_train_batch_size="32"
dropout_rate="0.2"
learning_rate="2e-05"
save_model=0
num_train_epochs=30
is_gen=0

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
    --num_train_epochs $num_train_epochs \
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
    --save_model $save_model \
    --is_gen $is_gen \
    --max_seq_len 512 \
    --fp16

: '
| 0              | 1                  | 2                        | 3             | 4            | 5      |
|:---------------|:-------------------|:-------------------------|:--------------|:-------------|:-------|
| Model Name     | Task Name          | per_gpu_train_batch_size | learning_rate | dropout_rate | is_gen |
| JABERv2-6L | abusive            | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | adult              | 16                       | 2e-05         | 0.2          | 0      |
| JABERv2-6L | age                | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | ans-claim          | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | dangerous          | 16                       | 7e-06         | 0.1          | 0      |
| JABERv2-6L | dialect-binary     | 32                       | 7e-06         | 0.2          | 0      |
| JABERv2-6L | dialect-region     | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | dialect-country    | 16                       | 7e-06         | 0.2          | 0      |
| JABERv2-6L | emotion            | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | emotion-reg        | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2-6L | gender             | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | hate-speech        | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | irony              | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | offensive          | 64                       | 2e-05         | 0.2          | 0      |
| JABERv2-6L | machine-generation | 32                       | 7e-06         | 0.1          | 0      |
| JABERv2-6L | sarcasm            | 8                        | 2e-05         | 0.2          | 0      |
| JABERv2-6L | sentiment          | 32                       | 7e-06         | 0.2          | 0      |
| JABERv2-6L | arabic-ner         | 8                        | 2e-05         | 0.2          | 0      |
| JABERv2-6L | aqmar-ner          | 8                        | 2e-05         | 0.2          | 0      |
| JABERv2-6L | dialect-pos        | 8                        | 2e-05         | 0.2          | 0      |
| JABERv2-6L | msa-pos            | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | ans-stance         | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | baly-stance        | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2-6L | xlni               | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2-6L | sts                | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | mq2q               | 16                       | 7e-06         | 0.1          | 0      |
| JABERv2-6L | topic              | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2-6L | wsd                | 16                       | 2e-05         | 0.1          | 0      |
| AT5Sv2    | abusive            | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | adult              | 16                       | 0.0001        | 0.1          | 1      |
| AT5Sv2    | age                | 8                        | 0.001         | 0.1          | 0      |
| AT5Sv2    | ans-claim          | 32                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | dangerous          | 8                        | 0.001         | 0.1          | 1      |
| AT5Sv2    | dialect-binary     | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | dialect-region     | 32                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | dialect-country    | 32                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | emotion            | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | emotion-reg        | 8                        | 0.001         | 0.1          | 0      |
| AT5Sv2    | gender             | 32                       | 0.0001        | 0.1          | 0      |
| AT5Sv2    | hate-speech        | 8                        | 0.001         | 0.2          | 0      |
| AT5Sv2    | irony              | 8                        | 0.001         | 0.1          | 0      |
| AT5Sv2    | offensive          | 64                       | 0.001         | 0.1          | 1      |
| AT5Sv2    | machine-generation | 32                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | sarcasm            | 32                       | 0.001         | 0.2          | 0      |
| AT5Sv2    | sentiment          | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | arabic-ner         | 16                       | 0.001         | 0.2          | 0      |
| AT5Sv2    | aqmar-ner          | 8                        | 0.001         | 0.1          | 0      |
| AT5Sv2    | dialect-pos        | 8                        | 0.001         | 0.2          | 0      |
| AT5Sv2    | msa-pos            | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | ans-stance         | 16                       | 0.001         | 0.1          | 1      |
| AT5Sv2    | baly-stance        | 32                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | xlni               | 16                       | 0.001         | 0.2          | 0      |
| AT5Sv2    | sts                | 8                        | 0.001         | 0.1          | 0      |
| AT5Sv2    | mq2q               | 64                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | topic              | 16                       | 0.001         | 0.1          | 0      |
| AT5Sv2    | wsd                | 16                       | 0.001         | 0.1          | 0      |
| SABER          | abusive            | 32                       | 2e-05         | 0.1          | 0      |
| SABER          | adult              | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | age                | 32                       | 7e-06         | 0.1          | 0      |
| SABER          | ans-claim          | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | dangerous          | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | dialect-binary     | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | dialect-region     | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | dialect-country    | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | emotion            | 32                       | 2e-05         | 0.1          | 0      |
| SABER          | emotion-reg        | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | gender             | 32                       | 7e-06         | 0.1          | 0      |
| SABER          | hate-speech        | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | irony              | 32                       | 2e-05         | 0.2          | 0      |
| SABER          | offensive          | 64                       | 2e-05         | 0.2          | 0      |
| SABER          | machine-generation | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | sarcasm            | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | sentiment          | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | arabic-ner         | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | aqmar-ner          | 32                       | 2e-05         | 0.1          | 0      |
| SABER          | dialect-pos        | 8                        | 2e-05         | 0.2          | 0      |
| SABER          | msa-pos            | 16                       | 7e-06         | 0.1          | 0      |
| SABER          | ans-stance         | 32                       | 2e-05         | 0.1          | 0      |
| SABER          | baly-stance        | 32                       | 7e-06         | 0.1          | 0      |
| SABER          | xlni               | 16                       | 2e-05         | 0.1          | 0      |
| SABER          | sts                | 16                       | 2e-05         | 0.2          | 0      |
| SABER          | mq2q               | 64                       | 2e-05         | 0.1          | 0      |
| SABER          | topic              | 32                       | 7e-06         | 0.2          | 0      |
| SABER          | wsd                | 16                       | 7e-06         | 0.2          | 0      |
| JABERv2   | abusive            | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | adult              | 32                       | 7e-06         | 0.2          | 0      |
| JABERv2   | age                | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2   | ans-claim          | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | dangerous          | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2   | dialect-binary     | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | dialect-region     | 32                       | 7e-06         | 0.1          | 0      |
| JABERv2   | dialect-country    | 32                       | 7e-06         | 0.2          | 0      |
| JABERv2   | emotion            | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | emotion-reg        | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | gender             | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | hate-speech        | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2   | irony              | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2   | offensive          | 64                       | 2e-05         | 0.1          | 0      |
| JABERv2   | machine-generation | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | sarcasm            | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | sentiment          | 32                       | 2e-05         | 0.2          | 0      |
| JABERv2   | arabic-ner         | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | aqmar-ner          | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2   | dialect-pos        | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2   | msa-pos            | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2   | ans-stance         | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | baly-stance        | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | xlni               | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | sts                | 8                        | 2e-05         | 0.1          | 0      |
| JABERv2   | mq2q               | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | topic              | 32                       | 2e-05         | 0.1          | 0      |
| JABERv2   | qa                 | 16                       | 2e-05         | 0.1          | 0      |
| JABERv2   | wsd                | 32                       | 2e-05         | 0.1          | 0      |
| AT5Bv2     | abusive            | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | adult              | 32                       | 0.001         | 0.1          | 1      |
| AT5Bv2     | age                | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | ans-claim          | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | dangerous          | 16                       | 0.001         | 0.2          | 0      |
| AT5Bv2     | dialect-binary     | 32                       | 0.0001        | 0.1          | 0      |
| AT5Bv2     | dialect-region     | 32                       | 0.0001        | 0.2          | 0      |
| AT5Bv2     | dialect-country    | 32                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | emotion            | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | emotion-reg        | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | gender             | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | hate-speech        | 32                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | irony              | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | offensive          | 64                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | machine-generation | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | sarcasm            | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | sentiment          | 32                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | arabic-ner         | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | aqmar-ner          | 8                        | 0.001         | 0.2          | 0      |
| AT5Bv2     | dialect-pos        | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | msa-pos            | 16                       | 0.0001        | 0.2          | 0      |
| AT5Bv2     | ans-stance         | 8                        | 0.001         | 0.1          | 0      |
| AT5Bv2     | baly-stance        | 8                        | 0.001         | 0.2          | 0      |
| AT5Bv2     | xlni               | 16                       | 0.001         | 0.2          | 1      |
| AT5Bv2     | sts                | 32                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | mq2q               | 32                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | topic              | 16                       | 0.001         | 0.1          | 0      |
| AT5Bv2     | qa                 | 16                       | 0.0001        | 0.2          | 0      |
| AT5Bv2     | wsd                | 32                       | 0.001         | 0.1          | 0      |
| JABER          | abusive            | 64                       | 2e-05         | 0.2          | 0      |
| JABER          | adult              | 16                       | 7e-06         | 0.2          | 0      |
| JABER          | age                | 32                       | 7e-06         | 0.1          | 0      |
| JABER          | ans-claim          | 8                        | 7e-06         | 0.1          | 0      |
| JABER          | dangerous          | 64                       | 2e-05         | 0.2          | 0      |
| JABER          | dialect-binary     | 16                       | 7e-06         | 0.2          | 0      |
| JABER          | dialect-region     | 32                       | 7e-06         | 0.1          | 0      |
| JABER          | dialect-country    | 16                       | 7e-06         | 0.2          | 0      |
| JABER          | emotion            | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | emotion-reg        | 16                       | 7e-06         | 0.1          | 0      |
| JABER          | gender             | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | hate-speech        | 8                        | 7e-06         | 0.1          | 0      |
| JABER          | irony              | 8                        | 2e-05         | 0.1          | 0      |
| JABER          | offensive          | 32                       | 2e-05         | 0.2          | 0      |
| JABER          | machine-generation | 32                       | 7e-06         | 0.1          | 0      |
| JABER          | sarcasm            | 32                       | 2e-05         | 0.1          | 0      |
| JABER          | sentiment          | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | arabic-ner         | 8                        | 7e-06         | 0.2          | 0      |
| JABER          | aqmar-ner          | 16                       | 2e-05         | 0.2          | 0      |
| JABER          | dialect-pos        | 8                        | 2e-05         | 0.1          | 0      |
| JABER          | msa-pos            | 16                       | 2e-05         | 0.2          | 0      |
| JABER          | ans-stance         | 32                       | 2e-05         | 0.1          | 0      |
| JABER          | baly-stance        | 32                       | 2e-05         | 0.1          | 0      |
| JABER          | xlni               | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | sts                | 8                        | 2e-05         | 0.1          | 0      |
| JABER          | mq2q               | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | topic              | 32                       | 2e-05         | 0.1          | 0      |
| JABER          | qa                 | 16                       | 2e-05         | 0.1          | 0      |
| JABER          | wsd                | 16                       | 2e-05         | 0.1          | 0      |
| AT5B           | abusive            | 8                        | 0.001         | 0.2          | 0      |
| AT5B           | adult              | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | age                | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | ans-claim          | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | dangerous          | 64                       | 0.001         | 0.1          | 0      |
| AT5B           | dialect-binary     | 32                       | 0.001         | 0.2          | 0      |
| AT5B           | dialect-region     | 16                       | 0.0001        | 0.1          | 0      |
| AT5B           | dialect-country    | 32                       | 0.001         | 0.2          | 0      |
| AT5B           | emotion            | 32                       | 0.001         | 0.2          | 1      |
| AT5B           | emotion-reg        | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | gender             | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | hate-speech        | 8                        | 0.001         | 0.2          | 0      |
| AT5B           | irony              | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | offensive          | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | machine-generation | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | sarcasm            | 64                       | 0.001         | 0.1          | 0      |
| AT5B           | sentiment          | 32                       | 0.001         | 0.2          | 0      |
| AT5B           | arabic-ner         | 32                       | 0.001         | 0.2          | 0      |
| AT5B           | aqmar-ner          | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | dialect-pos        | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | msa-pos            | 16                       | 0.001         | 0.1          | 0      |
| AT5B           | ans-stance         | 8                        | 0.001         | 0.1          | 0      |
| AT5B           | baly-stance        | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | xlni               | 8                        | 0.001         | 0.1          | 0      |
| AT5B           | sts                | 8                        | 0.001         | 0.2          | 0      |
| AT5B           | mq2q               | 16                       | 0.001         | 0.2          | 0      |
| AT5B           | topic              | 32                       | 0.001         | 0.1          | 1      |
| AT5B           | qa                 | 32                       | 0.001         | 0.1          | 0      |
| AT5B           | wsd                | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | abusive            | 16                       | 0.001         | 0.2          | 0      |
| AT5S           | adult              | 16                       | 0.001         | 0.2          | 0      |
| AT5S           | age                | 8                        | 0.001         | 0.1          | 0      |
| AT5S           | ans-claim          | 8                        | 0.001         | 0.1          | 0      |
| AT5S           | dangerous          | 64                       | 0.001         | 0.1          | 0      |
| AT5S           | dialect-binary     | 16                       | 0.0001        | 0.2          | 0      |
| AT5S           | dialect-region     | 16                       | 0.0001        | 0.1          | 0      |
| AT5S           | dialect-country    | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | emotion            | 32                       | 0.001         | 0.1          | 1      |
| AT5S           | emotion-reg        | 8                        | 0.001         | 0.1          | 0      |
| AT5S           | gender             | 16                       | 0.0001        | 0.1          | 0      |
| AT5S           | hate-speech        | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | irony              | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | offensive          | 64                       | 0.001         | 0.1          | 0      |
| AT5S           | machine-generation | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | sarcasm            | 32                       | 0.001         | 0.1          | 1      |
| AT5S           | sentiment          | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | arabic-ner         | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | aqmar-ner          | 8                        | 0.001         | 0.2          | 0      |
| AT5S           | dialect-pos        | 8                        | 0.001         | 0.1          | 0      |
| AT5S           | msa-pos            | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | ans-stance         | 8                        | 0.001         | 0.1          | 0      |
| AT5S           | baly-stance        | 32                       | 0.001         | 0.1          | 1      |
| AT5S           | xlni               | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | sts                | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | mq2q               | 32                       | 0.001         | 0.1          | 0      |
| AT5S           | topic              | 16                       | 0.001         | 0.1          | 0      |
| AT5S           | wsd                | 32                       | 0.001         | 0.1          | 0      |

'
