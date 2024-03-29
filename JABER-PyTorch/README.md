# JABER pytorch version

<p align="center">
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

<!-- -------------------------------------------------------------------------------- -->

* JABER (Junior Arabic BERt) is a 12-layer Arabic pretrained Language Model. 
We provide fine-tuning code for sentence classification tasks, 
which will allow you reproduce the test set submission that obtained rank one
 on [ALUE leaderboard](https://www.alue.org/leaderboard) at `01/09/2021`. 

* We also provide source code for fine tuning and models weights for `T5` models on ALUE
and some generative tasks. 

* **(01-01-2024)** We add support for 4 new models (JABERv2, JABERv2-6L, AT5Sv2, AT5Bv2), as well as 
code to support finetuning on [ORCA leaderboard](https://orca.dlnlp.ai/) tasks.
 
## Requirements
We recommend to create a conda environment 

```bash
conda create -n jaber_alue python=3.6.5
conda activate jaber_alue
```

* Run command below to install the environment

```bash
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
pip install -r envs/requirements.txt
```

## Downloads
 
### Models

* We provide pretrained models for:
    1. [JABER](https://huggingface.co/huawei-noah/JABER) Arabic BERT-base model.
    2. [AT5S](https://huggingface.co/huawei-noah/AT5S) Arabic T5-small model.
    3. [AT5B](https://huggingface.co/huawei-noah/AT5B) Arabic T5-base model.
    4. **(coming soon) Char-JABER** Arabic BERT-base model with Character level embeddings.
    5. **(coming soon) SABER** Arabic BERT-large model.
    6. [JABERv2](https://huggingface.co/huawei-noah/JABERv2) Arabic BERT-base model.
    7. [JABERv2-6L](https://huggingface.co/huawei-noah/JABERv2-6L) Arabic BERT-base 6 layer model.
    8. [AT5Sv2](https://huggingface.co/huawei-noah/AT5Sv2) Arabic T5-small model.
    9. [AT5Bv2](https://huggingface.co/huawei-noah/AT5Bv2) Arabic T5-base model.
     
* Place all downloaded models under `JABER-PyTorch/pretrained_models/` 

### External Modules

* For some necessary pre-processings we refer you to the ArabBERT code-base: https://github.com/aub-mind/arabert. In this regards, you can follow the steps given below: 
    1. Download the [preprocess.py](https://github.com/aub-mind/arabert/blob/master/preprocess.py). 
    2. Add the file under `/JABER-PyTorch`
    3. Comment the `ArabertPreprocessor` class in `generate_data.py`.
    4. Add `from preprocess import ArabertPreprocessor` in `generate_data.py`.


* To experiment on `EMD` dataset:
    1. Download [ArabicEmpatheticDialogues.py](https://github.com/aub-mind/Arabic-Empathetic-Chatbot/blob/master/model/ArabicEmpatheticDialogues.py)
    2. Add the file under `/JABER-PyTorch`
    3. Comment [line 25](https://github.com/aub-mind/Arabic-Empathetic-Chatbot/blob/b73f9b1151c26392bf45a6454b85283abb444300/model/ArabicEmpatheticDialogues.py#L25)
    4. Replace lines [80](https://github.com/aub-mind/Arabic-Empathetic-Chatbot/blob/b73f9b1151c26392bf45a6454b85283abb444300/model/ArabicEmpatheticDialogues.py#L80) [81](https://github.com/aub-mind/Arabic-Empathetic-Chatbot/blob/b73f9b1151c26392bf45a6454b85283abb444300/model/ArabicEmpatheticDialogues.py#L81)
    by:
 
 ```
"emotion": arabert_prep.preprocess(row[0]),
"context": arabert_prep.preprocess(row[1]),
-> 
"emotion": row[0],
"context": row[1],
```

* To evaluate on `QA` task:
    1. Download [eval_squad.py](https://github.com/UBC-NLP/araT5/blob/main/examples/eval_squad.py)
    2. Add the file under `/JABER-PyTorch`

* Please note that, our code will still run if you don't do the aforementioned step (the code will print a Warning)
but it will not produce the expected input data. 

### ALUE Data Download
  
1. Follow the instructions to download the ALUE datasets from 
[their official website](https://www.alue.org/tasks), and then place them under
`JABER-PyTorch/raw_datasets/`.
 
2. You may need to contact the authors of [ALUE](https://github.com/Alue-Benchmark/alue_baselines) in order to obtain
the correct train/dev/test split of `MDD` task.

3. You need to provide your own dev set for `MQ2Q` task, please follow these instructions:
      
      a. Download the English `QQP` dataset from [GLUE website](https://gluebenchmark.com/tasks).
      
      b. Randomly select 2k positive and negative samples (4k overall) from `dev` set.    
      
      c. Use an automatic translation service to translate the sentences to Arabic. 
      
      d. Create a file named `JABER-PyTorch/raw_datasets/mq2q.dev.tsv` where each line 
      contains one sample as follow: `lbl\tquestion_1\tquestion_2\n` (See the toy file 
      `JABER-PyTorch/raw_datasets/toy.mq2q.dev.tsv`)

### Generative Tasks Data Download

* Download the dataset for `QA` and `QG` tasks from:
    * [SQuAD_translate-train_squad.translate.train.en-ar.json](https://console.cloud.google.com/storage/browser/_details/xtreme_translations/SQuAD/translate-train/squad.translate.train.en-ar.json;tab=live_object)
    * [arcd.json](https://raw.githubusercontent.com/husseinmozannar/SOQAL/master/data/arcd.json)
    * [dev-context-ar-question-ar.json,test-context-ar-question-ar.json](https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip)
    * [tydiqa.goldp.ar.dev.json](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json) (extract only arabic data)
    * [tydiqa.goldp.ar.train.json](https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json) (extract only arabic data)

    and place them in `JABER-PyTorch/raw_datasets/QA/`

* Download the dataset for `TS` from:
    * [WikiLingua.pkl](https://drive.google.com/drive/folders/1PFvXUOsW_KSEzFm5ixB8J8BDB8zRRfHW?usp=sharing)
    * [EASC](https://sourceforge.net/projects/easc-corpus/files/EASC/EASC.zip/download)
    
    and place them in `JABER-PyTorch/raw_datasets/TS/`
    
## Process Data     

* We support the following tasks:
    * Classification: MDD, XNLI, OHSD, OOLD, FID, MQ2Q
    * Regression: SVREG
    * Muli-label Classification: SEC
    * NER: ANERCorp
    * Text2Text (generative for T5 only): TS, QA, QG, EMD

* We also support all ORCA tasks.
    
* Run this command to process ALUE or ORCA datasets:
     
```
export MODEL_NAME=JABER # JABERv2, JABERv2-6L, AT5S, AT5B, AT5Sv2, AT5Bv2
export BENCH_NAME=alue # orca
cd JABER-PyTorch

python generate_data.py --bench_name $BENCH_NAME --model_name $MODEL_NAME

```

## FineTuning 

* Run the script `run_alue.sh` to finetune a given model on a given task.

```
bash run_alue.sh
```

* You need change the following arguments in the scripts

```bash
model_name="JABER" # JABERv2, JABERv2-6L, AT5S, AT5B, AT5Sv2, AT5Bv2 
task_name="abusive" # any task from alue (8 tasks) or orca (29)
per_gpu_train_batch_size="32" # based on the best HP
dropout_rate="0.2" # based on the best HP
learning_rate="2e-05" # based on the best HP
save_model=0
num_train_epochs=30
is_gen=0 # based on the best HP
```

* The hyper-parameters for ORCA tasks is in a block comment at the bottom of the `run_alue.sh` script.

<!--
* To reproduce our the ALUE test submission you need to:
    1. For each of the ALUE 8 tasks, run the above command 5 times (random seeds is 
    automatically set by `--seed -1`). Don't forget to set the best HP.
    2. Run the following command to generate the `.tsv` test sets submission 
 to [ALUE leaderboard](https://www.alue.org/leaderboard):
 
```
export MODEL_NAME=JABER # or AT5S, AT5B
cd JABER-PyTorch
python generate_data.py --mode gather_alue --model_name $MODEL_NAME
```

* It will simply select, for each task, the test set predictions of the best model 
performing on its respective dev set. You will find 8 `.tsv` files in 
`JABER-PyTorch/alue_test_submission/${MODEL_NAME}`
 that you can directly submit to [ALUE leaderboard](https://www.alue.org/leaderboard).

-->


## Join the Huawei Noah's Ark community
 
* Main page: https://www.noahlab.com.hk/
* Github: https://github.com/huawei-noah

## License

This project's [license](LICENSE) is under the Apache 2.0 license.

## Citation

Please cite the following papers when using our code and model:

``` bibtex
@inproceedings{ghaddar2022revisiting,
  title={Revisiting Pre-trained Language Models and their Evaluation for Arabic Natural Language Processing},
  author={Ghaddar, Abbas and Wu, Yimeng and Bagga, Sunyam and Rashid, Ahmad and Bibi, Khalil and Rezagholizadeh, Mehdi and Xing, Chao and Wang, Yasheng and Duan, Xinyu and Wang, Zhefeng and others},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={3135--3151},
  year={2022}
}

@article{ghaddar2021jaber,
  title={Jaber and saber: Junior and senior arabic bert},
  author={Ghaddar, Abbas and Wu, Yimeng and Rashid, Ahmad and Bibi, Khalil and Rezagholizadeh, Mehdi and Xing, Chao and Wang, Yasheng and Xinyu, Duan and Wang, Zhefeng and Huai, Baoxing and others},
  journal={arXiv preprint arXiv:2112.04329},
  year={2021}
}

@article{ghaddar2024importance,
  title={On the importance of Data Scale in Pretraining Arabic Language Models},
  author={Ghaddar, Abbas and Langlais, Philippe and Rezagholizadeh, Mehdi and Chen, Boxing},
  journal={arXiv preprint arXiv:2401.07760},
  year={2024}
}
```



