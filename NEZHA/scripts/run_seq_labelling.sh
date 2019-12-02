# -*- coding: utf-8 -*-
##########
#Created on Fri Jul 12 11:05:22 2019
#start codes for running sequence labelling task.
#Note that read_tf_events.py is to read evaluation results from tf events file and do predict.
###########


CUDA_VISIBLE_DEVICES=0 python run_classifier_ner.py \
  --task_name=ner \
  --do_train=true \
  --do_eval=true \
  --do_train_and_eval=true \
  --data_dir=data/peoples-daily-ner \
  --save_checkpoints_steps=100 \
  --vocab_file=nezha/vocab.txt \
  --bert_config_file=nezha/bert_config.json \
  --init_checkpoint=nezha/model.ckpt-200 \
  --max_seq_length=256 \
  --train_batch_size=8 \
  --eval_batch_size=8 \
  --num_train_epochs=20 \
  --output_dir=output/peoples-daily-ner/
  
python read_tf_events.py \
  --task_name=ner \
  --task_data_dir=data/peoples-daily-ner \
  --max_seq_length=256 \
  --predict_batch_size=8 \
  --pretrained_model_dir=nezha/ \
  --task_output_dir=output/peoples-daily-ner/ \
  
