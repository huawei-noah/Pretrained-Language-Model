# JABER pytorch version

<p align="center">
  <img src="https://avatars.githubusercontent.com/u/12619994?s=200&v=4" width="150">
  <br />
  <br />
  <a href="LICENSE"><img alt="Apache License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" /></a>
</p>

<!-- -------------------------------------------------------------------------------- -->

JABER (Junior Arabic BERt) is a 12-layer Arabic pretrained Language Model. 
We only provide fine-tuning code for sentence classification tasks, 
which will allow you reproduce the test set submission that obtained rank one
 on [ALUE leaderboard](https://www.alue.org/leaderboard) at `01/09/2021`. 

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

## Download Dependencies

1. Download the pretrained model from [here](https://huggingface.co/huawei-noah/JABER) and then place it under 
`JABER-PyTorch/pretrained_models/`. 
  
2. Follow the instructions to download the ALUE datasets from 
[their official website](https://www.alue.org/tasks), and then place them under
`JABER-PyTorch/raw_datasets/`.
 
3. You may need to contact the authors of [ALUE](https://github.com/Alue-Benchmark/alue_baselines) in order to obtain
the correct train/dev/test split of `MDD` task.

4. You need to provide your own dev set for `MQ2Q` task, please follow these instructions:
      
      a. Download the English `QQP` dataset from [GLUE website](https://gluebenchmark.com/tasks).
      
      b. Randomly select 2k positive and negative samples (4k overall) from `dev` set.    
      
      c. Use an automatic translation service to translate the sentences to Arabic. 
      
      d. Create a file named `JABER-PyTorch/raw_datasets/mq2q.dev.tsv` where each line 
      contains one sample as follow: `lbl\tquestion_1\tquestion_2\n` (See the toy file 
      `JABER-PyTorch/raw_datasets/toy.mq2q.dev.tsv`)

## Process Data     

* For some necessary pre-processings we refer you to the ArabBERT code-base: https://github.com/aub-mind/arabert. In this regards, you can follow the steps given below: 
    1. Download the [preprocess.py](https://github.com/aub-mind/arabert/blob/master/preprocess.py). 
    2. Add the file under `/JABER-PyTorch`
    3. Comment the `ArabertPreprocessor` class in `generate_data.py`.
    4. Add `from preprocess import ArabertPreprocessor` in `generate_data.py`.


* Please note that, our code will still run if you don't do the aforementioned step (the code will print a Warning)
but it will not produce the expected input data. 
        

* Run this command to process ALUE datasets:
 
```
cd JABER-PyTorch
python generate_data.py --mode train 
```

* Please check that directory\file names match those in `process_alue()`  method in 
`generate_data.py`.

## ALUE FineTuning 

* The following command will finetune **JABER** 5 times for a given ALUE task 
(FID in this demo):

```bash
#export CUDA_VISIBLE_DEVICES=0 # the ID of GPU to run the experiments on
export TASK=fid # mq2q | oold | ohsd | svreg | sec  | fid  | xnli | mdd
bash run_alue.sh $TASK
```

* The above code will automatically run `run_alue.py` five times for each task
 using different random seeds (`--seed -1`). 
 
* To reproduce our test submission you need to finetune **JABER** on all tasks 
(40 experiments in total). 
 
* This would generate 40 `./alue_predictions/jaber.{TASK}.{max_dev_score}.pkl` files,
 which each contains the test set predictions for the best performing checkpoint
  on its respective dev set. 
  
* Here are the hyper-parameters we used to generate test files for 
[ALUE leaderboard](https://www.alue.org/leaderboard):

| hp                  | MQ2Q | OOLD | OHSD | SVREG | SEC  | FID  | XNLI | MDD  |
|---------------------|------|------|------|-------|------|------|------|------|
| batch_size          | 64   | 128  | 32   | 8     | 16   | 32   | 16   | 32   |
| lr                  | 2e-5 | 2e-5 | 7e-6 | 2e-5  | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| hidden_dropout_prob | 0.3  | 0.2  | 0.3  | 0.1   | 0.1  | 0.1  | 0.1  | 0.2  |

* However, we already entered these configurations in `run_alue.sh`.

* Finally, run the following command to generate the `.tsv` test sets submission 
 to [ALUE leaderboard](https://www.alue.org/leaderboard). 
   
```
cd JABER-PyTorch
python generate_data.py --mode test 
```

* It will simply select, for each task, the test set predictions of the best model 
performing on its respective dev set. You will find 8 `.tsv` files in `JABER-PyTorch/alue_test_submission`
 that you can directly submit to [ALUE leaderboard](https://www.alue.org/leaderboard).

## Join the Huawei Noah's Ark community
 
* Main page: https://www.noahlab.com.hk/
* Github: https://github.com/huawei-noah

## License

This project's [license](LICENSE) is under the Apache 2.0 license.

## Citation

Please cite the following [paper](https://arxiv.org/abs/2112.04329) when using our code and model:

``` bibtex
@misc{ghaddar2021jaber,
      title={JABER: Junior Arabic BERt}, 
      author={Abbas Ghaddar and Yimeng Wu and Ahmad Rashid and Khalil Bibi and Mehdi Rezagholizadeh and Chao Xing and Yasheng Wang and Duan Xinyu and Zhefeng Wang and Baoxing Huai and Xin Jiang and Qun Liu and Philippe Langlais},
      year={2021},
      eprint={2112.04329},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



