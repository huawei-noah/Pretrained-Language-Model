# Step 2: Apply ternary weight splitting and finetune BinaryBERT.
# Tips:
# 1. If trained with data augmentation, please add --aug_train
# 2. For activation quantziation, use uniform quant for A8, with ACT2FN=gelu;
#    use lsq quant for A4, use lsq, with ACT2FN=relu to ensure non-negativity of LSQ asymmetric quantization

TASK=$1
DATA_DIR=$2
TEACHER_MODEL_DIR=$3
STUDENT_MODEL_DIR=$4
wbits=$5
abits=$6
JOB_ID=Ternary_W${wbits}A${abits}
echo $TASK
echo $DATA_DIR
echo $TEACHER_MODEL_DIR
echo $STUDENT_MODEL_DIR
echo $wbits
echo $abits
echo $JOB_ID

if [ $abits == 4 ]
then
act_quan_method=lsq
ACT2FN=relu
else
act_quan_method=uniform
ACT2FN=gelu
fi

export CUDA_VISIBLE_DEVICES=7
if [ $TASK == 1 ]
then
TASK_NAME=SQuADv1.1
python quant_task_distill_squad.py \
    --data_dir ${DATA_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --eval_step 1000 \
    --num_train_epochs 1 \
    --ACT2FN ${ACT2FN} \
    --output_dir output/${JOB_ID}/${TASK_NAME} \
    --kd_type two_stage \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method bwn \
    --input_bits ${abits} \
    --input_quant_method ${act_quan_method} \
    --clip_lr 1e-3 \
    --learnable_scaling \
    --is_binarybert \
    --split
else
TASK_NAME=SQuADv2.0
python quant_task_distill_squad.py \
    --data_dir ${DATA_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --eval_step 1000 \
    --num_train_epochs 1 \
    --ACT2FN gelu \
    --output_dir output/${JOB_ID}/${TASK_NAME} \
    --kd_type two_stage \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method bwn \
    --input_bits ${abits} \
    --input_quant_method uniform \
    --clip_lr 1e-3 \
    --learnable_scaling \
    --version_2_with_negative \
    --is_binarybert \
    --split
fi

