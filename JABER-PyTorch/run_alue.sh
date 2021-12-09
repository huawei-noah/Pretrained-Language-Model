#!/bin/sh

TASK=$1

if [ ${TASK,,} == mq2q ]; then
    DROPOUT=0.3
    LR=2e-5
    BATCH_SIZE=64
elif [ ${TASK,,} == oold ]; then
    DROPOUT=0.2
    LR=2e-5
    BATCH_SIZE=128
elif [ ${TASK,,} == ohsd ]; then
    DROPOUT=0.3
    LR=7e-6
    BATCH_SIZE=32
elif [ ${TASK,,} == svreg ]; then
    DROPOUT=0.1
    LR=2e-5
    BATCH_SIZE=8
elif [ ${TASK,,} == sec ]; then
    DROPOUT=0.1
    LR=2e-5
    BATCH_SIZE=16
elif [ ${TASK,,} == fid ]; then
    DROPOUT=0.1
    LR=2e-5
    BATCH_SIZE=32
elif [ ${TASK,,} == xnli ]; then
    DROPOUT=0.1
    LR=2e-5
    BATCH_SIZE=16
elif [ ${TASK,,} == mdd ]; then
    DROPOUT=0.2
    LR=2e-5
    BATCH_SIZE=32
else
    echo "{$TASK} is not supported, Exit!!!!"
    exit 125
fi

export MAX_SEQ_LEN=128 
export PKL_DATA=./raw_datasets/alue.$MAX_SEQ_LEN.jaber.pkl
export MODEL_PATH=./pretrained_models/JABER
export MODEL_NAME=pytorch_model.bin
export OUTPUT_DIR=fine_tune_result/jaber/$TASK
export RESULT_FILE_NAME=alue.log

for i in 1 2 3 4 5
do
    python run_alue.py \
        --data $PKL_DATA \
        --model_path $MODEL_PATH \
        --model_name $MODEL_NAME \
        --task_name $TASK \
        --output_dir $OUTPUT_DIR \
        --max_seq_length $MAX_SEQ_LEN \
        --do_train \
        --learning_rate $LR \
        --weight_decay 0.01 \
        --num_train_epochs 30 \
        --warmup_portion 0.1 \
        --seed -1 \
        --evaluate_during_training \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --gradient_accumulation_steps 1 \
        --log_name $RESULT_FILE_NAME \
        --hidden_dropout_prob $DROPOUT \
        --overwrite_output_dir \
        --save_model False \
        --do_eval 
done