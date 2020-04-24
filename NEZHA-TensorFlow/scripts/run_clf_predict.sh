# -*- coding: utf-8 -*-
#####################
#Created on Fri Jul 12 11:05:22 2019
#start codes for running clf(lxqmc/chnsenti/xnli) predict tasks
#####################

python ../run_classifier.py \
  --task_name=$1 \
  --do_predict=true \
  --data_dir=$2 \
  --vocab_file=$3/vocab.txt \
  --bert_config_file=$3/bert_config.json \
  --init_checkpoint=$4 \
  --max_seq_length=$5 \
  --predict_batch_size=$6 \
  --output_dir=$7/
