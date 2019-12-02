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


## 3 Finetune BERT

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


## 4. NEZHA model download

Models [download](https://pan.baidu.com/s/1V7btNIDqBHvz4g9LOPLeeg), password: x3qk

* We released 4 Chinese pretrained models, `bert-base` and `bert-large`, models with `WWM` tag means `Whole Word Masking`.

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