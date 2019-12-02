# -*- coding: utf-8 -*-
############
#Created on Fri Jul 12 11:05:22 2019
#start codes for running squad-like MRC task
#############


CUDA_VISIBLE_DEVICES=0 python run_squad.py \
  --vocab_file=nezha/vocab.txt \
  --bert_config_file=nezha/bert_config.json \
  --init_checkpoint=nezha/model.ckpt-200 \
  --do_train=True \
  --train_file=data/cmrc/new_cmrc2018_train.json \
  --do_predict=True \
  --predict_file=data/cmrc/new_cmrc2018_dev.json \
  --train_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=0.05 \
  --max_seq_length=512 \
  --doc_stride=128 \
  --do_lower_case=False \
  --output_dir=output/cmrc/

python cmrc2018_evaluate.py data/new_cmrc2018_dev.json output/cmrc/dev_predictions.json output/cmrc/result_metric.txt
