# -*- coding: utf-8 -*-
#################
#Created on Fri Jul 12 11:05:22 2019
#start codes for running clf(lcqmc/chnsenti/xnli) tasks.
# task_name is 'lcqmc' for lcqmc task, 'xnli' for xnli task and 'text_clf' for chnsenti
#read_tf_events is to find the best eval ckpt and do predict
##################

CUDA_VISIBLE_DEVICES=1 python ../run_classifier.py \
  --task_name=lcqmc \
  --do_train=true \
  --do_eval=true \
  --do_train_and_eval=true \
  --data_dir=../data/lcqmc/ \
  --save_checkpoints_steps=50 \
  --vocab_file=../nezha/vocab.txt \
  --bert_config_file=../nezha/bert_config.json \
  --init_checkpoint=../nezha/model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --num_train_epochs=5 \
  --output_dir=../output/lcqmc/

python ../read_tf_events.py \
  --task_name=lcqmc \
  --task_data_dir=../data/lcqmc/ \
  --max_seq_length=128 \
  --predict_batch_size=32 \
  --pretrained_model_dir=../nezha/ \
  --task_output_dir=../output/lcqmc/ \
