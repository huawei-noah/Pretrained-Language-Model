# NEZHA

* NEZHA (NEural contextualiZed representation for CHinese lAnguage understanding) is the Chinese pretrained language model currently based on BERT developed by Huawei Noah's Ark lab.
* Please note that this code is for training NEZHA on normal GPU clusters, and not identical to what we used in training NEZHA [ModelArts](https://www.huaweicloud.com/product/modelarts.html) provided by Huawei Cloud.
* For the convenience of reproducing our result, this code is revised based on the early versions of NVIDIA's [code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Google's [code](https://github.com/google-research/bert), with integrating all techniques we adopted. 


## 1. Prepare data

Following the data preparation as in BERT, run command as below:

```shell
python utils/create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=./your/path/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
 ```

## 2. Pretrain

First, prepare the horovod distributed training environment, and then run **scripts/run_pretraining.sh**:


## 3 Finetune NEZHA

For the time being, we support three kinds of fine-tuning tasks: text classification, sequence labelling, and SQuAD-like MRC.  
Our fine-tuning codes are mainly based on [Google BERT](https://github.com/google-research/bert),[BERT NER](https://github.com/ProHiryu/bert-chinese-ner) ,[CMRC2018-DRCD-BERT](https://github.com/johndpope/CMRC2018-DRCD-BERT)

- Download the pretrained model and unpack model file,vocab file and config file in `nezha/`.
- Build the fine-tuning task  
(1)**scripts/run_clf.sh** is for text classification tasks such as LCQMC,ChnSenti,XNLI.  
(2)**scripts/run_seq_labelling.sh** is for sequence labelling tasks such as Peoples-daily-NER .  
(3)**scripts/run_reading.sh** is for SQuAD-like MRC tasks such as CMRC2018(https://github.com/ymcui/cmrc2018).  
- Get the evaluation and test results from the related output repository.
- Note that CMRC task evaluation is a little bit different. Please run this script separately:
```shell
python cmrc2018_evaluate.py data/cmrc/cmrc2018_dev.json output/cmrc/dev_predictions.json output/cmrc/metric.txt. 
```
cmrc2018_evaluate.py can be found [here](https://github.com/ymcui/cmrc2018/tree/master/baseline).

## 4. NEZHA model download

 We released 4 Chinese pretrained models, `NEZHA-base` and `NEZHA-large`, models with `WWM` tag means `Whole Word Masking`.

* NEZHA-base: 
Baidu Yun [download](https://pan.baidu.com/s/1UVQjy9v_Sv4cQd1ELdjqww), password:ntn3; 
Google Driver [download](https://drive.google.com/drive/folders/1tFs-wMoXIY8zganI2hQgDBoDPqA8pSmh?usp=sharing)

* NEZHA-base-WWM: 
Baidu Yun [download](https://pan.baidu.com/s/1-YG8e5V2zKCnR3azsGZT1w), password:f68o; 
Google Driver [download](https://drive.google.com/drive/folders/1bK6WbqAG-B6BX2d9RPprnh2MPK6zL0t_?usp=sharing)

* NEZHA-large: 
Baidu Yun [download](https://pan.baidu.com/s/1R1Ew-Lu8oIP6QhWO6nqp5Q), password:7thu; 
Google Driver [download](https://drive.google.com/drive/folders/1ZPPM5XtTTOrS_CDRak1t2nCBU-LFZ_zs?usp=sharing)

* NEZHA-large-WWM:
Baidu Yun [download](https://pan.baidu.com/s/1JK1RLIJd2wpuypku3stt8w), password:ni4o; 
Google Driver [download](https://drive.google.com/drive/folders/1LOAUc9LXyogC2gmP_q1ojqj41Ez01aga?usp=sharing)

* MD5 File:
Baidu Yun [download](https://pan.baidu.com/s/1EeFXcmFBaJ3tDQQGYoaQPQ ), password:yxpk;  
Google Driver [download](https://drive.google.com/file/d/1eWRvd3k5XK6sOlAPPpCWowHubLeRk3G-/view?usp=sharing)

We further released a multilingual pretrained model `NEZHA-base-multilingual-11-cased` tokenized with [Byte BPE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BBPE). Currently the model covers 11 languages (in alphabetical order): Arabic, Deutsch, English, Espanol, French, Italian, Malay Polish, Portuguese, Russian and Thai. Please use the tokenizationBBPE.py in [Byte BPE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BBPE) as the tokenizer (i.e., replace the original tokenization.py with this tokenizationBBPE.py) if you want to use our multilingual pretrained NEZHA model. 

* NEZHA-base-multilingual-11-cased: 
Baidu Yun [download](https://pan.baidu.com/s/1yddjyoMxIKPfsUvx1YDH-Q), password:gs31; 
Google Driver [download](https://drive.google.com/file/d/1430BzQbiHxvq8ZrKj6FQu3nmDMLtKmDe/view?usp=sharing)


## 5. References

> Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, 
> Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu.
> [NEZHA: Neural Contextualized Representation for Chinese Language Understanding.](https://arxiv.org/abs/1909.00204)
> arXiv preprint arXiv:1909.00204

```
@article{wei2019nezha,
  title = {NEZHA: Neural Contextualized Representation for Chinese Language Understanding},
  author = {Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu},  
  journal = {arXiv preprint arXiv:1909.00204},
  year = {2019},
}
```
