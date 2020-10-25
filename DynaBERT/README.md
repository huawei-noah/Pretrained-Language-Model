# DynaBERT

* DynaBERT can flexibly adjust the size and latency by selecting adaptive width and depth, and 
the subnetworks of it have competitive performances as other similar-sized compressed models.
The training process of DynaBERT includes first training a width-adaptive BERT and then 
allowing both adaptive width and depth using knowledge distillation. 
The overview of DynaBERT learning is shown below. 
<img src="dynabert_overview.png" width="800" height="320"/>
<br />


* This code is modified based on the repository developed by Hugging Face: [Transformers v2.1.1](https://github.com/huggingface/transformers/tree/v2.1.1)
* The results in the paper are produced by using single V100 GPU.

## Environment and data

- Prepare environment.
```bash
pip install -r requirements.txt
```

- Prepare the data sets.
The **original** GLUE data can be accessed from [here](https://gluebenchmark.com/tasks).
The **augmented data** for each task can be 
generated using the method in  [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) using the script from 
[this repository](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT).
Put the original data (`train.csv`, `dev.csv`) and the augmented data (named as `train_${TASK_NAME}_aug_with_logits.csv`)
to:
```
${GLUE_DIR}/${TASK_NAME}
```


## Train DynaBERT on GLUE tasks

**Step 1.** 
Fine-tune the pretrained [BERT base model](https://huggingface.co/bert-base-uncased).
Put the fine-tuned models (including `config.json`, `pytorch_model.bin`, `vocab.txt`) to: 
```
${OUTPUT_DIR}/${TASK_NAME}/bert/best
```



**Step 2.** Train DynaBERTw.
Run `run_glue.py` to train DynaEBRTw by setting `training_phase` to `dynabertw`.
We use the fine-tuned BERT model in Step 1 as the fixed teacher model, and to initialize the width-adaptive DynaBERTw.
```
python run_glue.py \
	--model_type bert \
	--task_name ${TASK_NAME} \
	--do_train \
	--data_dir ${GLUE_DIR}/${TASK_NAME}/ \
	--model_dir ${FINETUNED_MODEL_DIR} \
	--output_dir ${DYNABERTW_DIR} \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32 \
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--width_lambda1 1.0 \
	--width_lambda2 0.1 \
	--training_phase dynabertw \
	--data_aug
```

**Step 3.** Train DynaBERT.
Run `run_glue.py` to train DynaEBRTw by setting `training_phase` to `dynabert`.
We use the trained DynaBERTw model from Step 2 as the fixed teacher model, and to initialize the both width- and depth-adaptive DynaBERT.
```
python run_glue.py \
	--model_type bert \
	--task_name ${TASK_NAME} \
	--do_train \
	--data_dir ${GLUE_DIR}/${TASK_NAME}/ \
	--model_dir ${DYNABERTW_DIR} \
	--output_dir ${DYNABERT_DIR} \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32 \
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--depth_lambda1 1.0 \
	--depth_lambda2 1.0 \
	--training_phase dynabert \
	--data_aug
```

**Step 4. (optional)**  Run `run_glue.py` for optional final fine-tuning by setting `training_phase` to `final_finetuning`.
We use the trained DynaBERT model from Step 3 for initialization.
```
python run_glue.py \
	--model_type bert \
	--task_name ${TASK_NAME} \
	--do_train \
	--data_dir ${GLUE_DIR}/${TASK_NAME}/ \
	--model_dir ${DYNABERT_DIR} \
	--output_dir ${FINAL_FINETUNING_DIR} \
	--max_seq_length 128 \
	--learning_rate 2e-5 \
	--per_gpu_train_batch_size 32\
	--num_train_epochs 3 \
	--width_mult_list 0.25,0.5,0.75,1.0 \
	--depth_mult_list 0.5,0.75,1.0 \
	--logging_steps 10 \
	--training_phase final_finetuning 
```


## Trained DynaBERT download and inference
- The trained DynaBERT on GLUE tasks can be downloaded from [here](https://drive.google.com/file/d/1pYApaDcse5QIB6lZagWO0uElAavFazpA/view?usp=sharing).

- Run `eval_glue.py` to evaluate a sub-network of DynaBERT by specifying `width_multiplier` and `depth_multiplier`.
```
python eval_glue.py \
	--model_type bert \
	--task_name ${TASK_NAME} \
	--data_dir ${GLUE_DIR}/${TASK_NAME}/ \
	--max_seq_length 128 \
	--model_dir ${TRAINED_DYNABERT_DIR} \
	--output_dir ${OUTPUT_DIR} \
	--depth_mult 0.75 \
	--width_mult 1.0 
```

## To Dos
- Support Horovod for efficient distributed learning.
- Use the proposed DynaBERT in the pre-training phase.


## Reference
Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu.
[DynaBERT: Dynamic BERT with Adaptive Width and Depth](https://arxiv.org/abs/2004.04037).
```
@inproceedings{hou2020dynabert,
  title = {DynaBERT: Dynamic BERT with Adaptive Width and Depth},
  author = {Lu Hou, Zhiqi Huang, Lifeng Shang, Xin Jiang, Xiao Chen, Qun Liu},  
  booktitle={Advances in Neural Information Processing Systems}
  year = {2020},
}
```