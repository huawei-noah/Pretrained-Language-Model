#########################################################################
# run_classifier.sh for sentence classification task
#########################################################################
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run_sequence_classifier.py \
  --task_name=text-clf \
  --do_train \
  --do_eval \
  --data_dir=data/chnsenti/ \
  --bert_model=pretrained_models/nezha-cn-base/ \
  --max_seq_length=128 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=3e-5  \
  --num_train_epochs=10.0 \
  --output_dir=output/0414chnsenti/
