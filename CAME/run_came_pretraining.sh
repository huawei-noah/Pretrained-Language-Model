#!/bin/bash

# num_eval_examples too large will cause 'unable to write to file' https://github.com/pytorch/pytorch/issues/2926

python -m torch.distributed.launch --nproc_per_node=8 \
    /home/ma-user/work/Old_BERT/run_pretraining.py \
    --seed=12439 \
    --do_train \
    --do_eval \
		--optimizer=came \
    --config_file=/home/ma-user/work/Old_BERT/bert_large_config.json \
    --output_dir=/cache/results \
     --fp16 \
    --allreduce_post_accumulation \
    --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=256 \
	 --bert_model=bert-large-uncased \
    --log_freq=1 \
    --train_batch_size=4096 \
	--dev_batch_size=64 \
    --learning_rate=0.00024 \
    --warmup_proportion=0.2 \
	 --num_steps_per_checkpoint=5 \
    --input_dir=/cache/data/train_data \
    --dev_dir=/cache/data/dev_data \
    --phase2 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --max_steps=20000 \
    --init_checkpoint=None \
    --phase1_end_step=0 

