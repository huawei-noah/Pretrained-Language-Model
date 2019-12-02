# -*- coding: utf-8 -*-
#################
#Run pretraining.
##################


mpiexec --allow-run-as-root --bind-to socket -np 2 python run_pretraining.py \
  --input_file=./data/pretrain-toy/*.tfrecord \
  --output_dir=./nezha/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./nezha/bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=200 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --horovod  
