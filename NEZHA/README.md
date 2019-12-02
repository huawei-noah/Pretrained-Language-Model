# NEZHA

* NEZHA is the Chinese pretrained language models developed by Huawei Noah's Ark lab.
* Please notice that this code is not the code we use, we train NEZHA on Huawei Cloud ModelArts.
* For convenience of reproducing our result, this code is revised based on one early version of NVIDIA [code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/LanguageModeling/BERT) and Google [code](https://github.com/google-research/bert), with integrating all techniques we adopted. 


## 1. Prepare data

Follow data preparation as bert, run command like below:

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

First prepare horovod distributed training environment. Then run **scripts/run_pretraining.sh**:


## 3 Finetune BERT

now we support 3 kinds of tasks fine-tuning: text classification,sequence labelling,SQuAD-like MRC.  
our fine-tuning codes are mainly based on [Google BERT](https://github.com/google-research/bert),[BERT NER](https://github.com/ProHiryu/bert-chinese-ner) ,[CMRC2018-DRCD-BERT](https://github.com/johndpope/CMRC2018-DRCD-BERT)

- Download pretrained model and unpack model file,vocab file and config file in `nezha/`.
- Build fine-tuning task  
(1)**scripts/run_clf.sh** is for text classification tasks like LCQMC,ChnSenti,XNLI.  
(2)**scripts/run_seq_labelling.sh** is for sequence labelling tasks like Peoples-daily-NER .  
(3)**scripts/run_reading.sh** is for SQuAD-like MRC tasks like CMRC2018(https://github.com/ymcui/cmrc2018).  
- Get evaluation and test results from related output repository.
- Note that cmrc task evaluation is a little bit different. please run this script separately:
```shell
python cmrc2018_evaluate.py data/cmrc/cmrc2018_dev.json output/cmrc/dev_predictions.json output/cmrc/metric.txt. 
```


## 4. Pretrained NEZHA download

[Donwload](https://pan.baidu.com/s/1V7btNIDqBHvz4g9LOPLeeg), password: x3qk

* We released 4 Chinese pretrained models, `bert-base` and `bert-large`, with `WWM` tag means `Whole Word Masking`.

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