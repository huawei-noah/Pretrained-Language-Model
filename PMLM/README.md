
# Probabilistically Masked Language Model ([paper](https://arxiv.org/pdf/2004.11579.pdf))([slides](https://drive.google.com/file/d/1F5T5bvlvBLTNptUshvw6TzbVzB7gzBr8/view)).
**PMLM** is an improved method for pretrained language model. Trained without the complex two-stream self-attention, PMLM can be treated as a simple approximation of XLNet.
**PMLM** bridges the gap between the autoregressive language model (e.g. GPT) and the masked language model (e.g. BERT). 
On one hand, PMLM can generate fluent text in arbitrary text order. On the other hand, PMLM consistently and significantly 
outperforms BERT on natural language understanding tasks. 

## LASTEST NEWS
We have added the support for text generation with PMLM. (30/04/2021 updated)

## Requirements
* Python 3
* [TensorFlow](https://www.tensorflow.org/) 1.13

## Usage for pretraining and finetuning
To minimize the effort you need to get familiar with PMLM as well as to reproduce our results, we provide our code as compatible with the repository [BERT](https://github.com/google-research/bert). 

Compared with [BERT](https://github.com/google-research/bert), our changes are only in three files. One is ```create_pretraining_data.py```, the second is ```bert_config.json```, and the third is ```modeling.py```. 
u-PMLM-R(A) can be pretrained and finetuned with [BERT](https://github.com/google-research/bert) framework by replacing these three files with ours. 
You may also convert it into a Pytorch version and work with [Hugging Face](https://huggingface.co/) framework.  Below are some descriptions of the files we provide. 
* ```ceate_pretraining_data.py``` The differences from the preprocessing scripts are in line 369-372. The argument ```masked_lm_prob``` is redefined as the upper bound of the probability. 
For example, when masked_lm_prob=0.8, the uniform distribution takes values within [0, 0.8]. When masked_lm_prob=1/6, it approximates XLNet.
* ```bert_config.json``` We add a boolean field ```use_relative_position```. use_relative_position=True corresponds to U-PMLM-R and use_relative_position=False corresponds to U-PMLM-A.
* ```modeling.py``` You may pay attention to the places where the keyword ```use_relative_position``` occurs.


## Released Models
We release two models for downloading: [u-PMLM-R](https://drive.google.com/file/d/1SdytT4TQVIOUcbCQfGSSA067mde0ytA3/view?usp=sharing) and [u-PMLM-A](https://drive.google.com/file/d/15yBa896-1RpsJJB8mKCmy14Xi8gRFAii/view?usp=sharing). Both models are trained on Wikipedia and BookCorpus. Both models use the cased version of BERT vocabulary and the uniform distribution takes values in [0.1.0]. 
We observe that there are no much differences regarding performances when trained with uniform distribution valued in [0,0.5].  
Note that the sequence length is set to 128 rather than 512, hence the embedding for positions 128~511 are not trained for u-PMLM-A, which might harm tasks requiring more than 128 tokens per sequence such as SQuAD. This is not a problem for U-PMLM-R though, as it uses relative position.

## Employing PMLM for arbitrily ordered text generation (30/04/2021 updated)
We release the PMLM  model that has been finetuned on the wikitext-103 dataset. The model can be downloded [here](https://drive.google.com/file/d/1jRcBO4wQiR_eZW3V58iCBn18MgXz4oLz/view?usp=sharing). Use the command ```python interactive_conditional_samples_sincos_acrostic``` for text generation.

Note: The code can also load the pretrained PMLM models such as u-PMLM-A and u-PMLM-R. However, the pretrained models are trained with BERT-style data, where 50% of the training sequences are composed of two sentences that are not adjacent. Thus the generated sentences are not coherent at some position in the middle. The PMLM model finetuned with wikitext-103, however, does not have such problem as the training sequence are coherent sentences in the corpus.


## Detailed  experimental results on dev set of GLUE for u-PMLM-R
In our original paper, we did not search for a good hyperparameter setting on GLUE tasks for u-PMLM. We just followed the settings of BERT. 
Later we learned that the finetune process is unstable and is sensitive to the hyperparameter settings. So we conduct a simple parameter search.

For each task, we conduct eight initial finetune experiments with four different learning rates and two training epoch number. 
For learning rate, we mostly search among (1e-5, 2e-5, 3e-5, 4e-5). For epoch number, we roughly follow our experience in finetuning our large model NEZHA
(NEZHA ranked 3rd on the SuperGLUE leaderboard). After we find the best combination of learning rate and epoch number among the eight configurations for each task, 
we conduct eight finetune experiments with the best configuration using randomized random_seed for each task. The full results of the eight runs for each task and the corresponding results are shown below.

| Number of Runs              | CoLA     |          | SST-2    |          | MRPC     |         | STS-B    |        | QQP      |          |
|-----------------------------|----------|----------|----------|----------|----------|---------|----------|--------|----------|----------|
| 1                           | 63.42    | 62.13    | 93.35    | 92.89    | 88.24    | 87.75   | 90.58    | 90.58  | 91.38    | 91.25    |
| 2                           | 63.57    | 61.61    | 93.81    | 93.23    | 87.75    | 87.75   | 90.46    | 90.35  | 91.55    | 91.52    |
| 3                           | 64.22    | 62.64    | 93.46    | 92.89    | 87.01    | 86.76   | 91.14    | 91.14  | 91.33    | 91.33    |
| 4                           | 62.97    | 60.58    | 93.58    | 92.66    | 87.5     | 87.5    | 90.1     | 90.06  | 91.34    | 91.29    |
| 5                           | 63.16    | 62.6     | 93.46    | 92.55    | 88.24    | 87.99   | 90.51    | 90.19  | 91.49    | 91.45    |
| 6                           | 61.83    | 59.06    | 93.69    | 93.34    | 87.75    | 87.5    | 90.94    | 90.92  | 91.39    | 91.36    |
| 7                           | 63.79    | 62.57    | 94.27    | 94.15    | 87.25    | 86.76   | 90.7     | 90.69  | 91.34    | 91.33    |
| 8                           | 62.34    | 62.34    | 93.69    | 93.46    | 88.24    | 87.25   | 90.22    | 89.87  | 91.42    | 91.38    |
| Average                     | 63.1625  | 61.69125 | 93.66375 | 93.14625 | 87.7475  | 87.4075 | 90.58125 | 90.475 | 91.405   | 91.36375 |
| Best                        | 64.22    | 62.64    | 94.27    | 94.15    | 88.24    | 87.99   | 91.14    | 91.14  | 91.55    | 91.52    |
| Worst                       | 61.83    | 59.06    | 93.35    | 92.55    | 87.01    | 86.76   | 90.1     | 89.87  | 91.33    | 91.25    |
| Epochs                      | 10       |          | 5        |          | 4        |         | 9        |        | 10       |          |
| Learning Rate               | 5.00E-06 |          | 2.00E-05 |          | 2.00E-05 |         | 2.00E-05 |        | 2.00E-05 |          |
| Evaluate every (steps)            | 100      |          | 200      |          | 100      |         | 100      |        | 3000     |          |
| Total number of evaluations | 28       |          | 54       |          | 6        |         | 18       |        | 39       |          |
| Batch size                  | 32       |          | 32       |          | 32       |         | 32       |        | 32       |          |

| Number of Runs              | MNLI-m   |       | MNLI-mm  |          | QNLI     |          | RTE             |                |
|-----------------------------|----------|-------|----------|----------|----------|----------|-----------------|----------------|
| 1                           | 85.46    | 85.43 | 85.91    | 85.8     | 92.73    | 92.53    | 77.26           | 75.81          |
| 2                           | 85.02    | 84.83 | 86.1     | 85.78    | 92.7     | 92.6     | 72.92           | 72.56          |
| 3                           | 85.23    | 84.84 | 85.49    | 85.37    | 92.92    | 92.92    | 78              | 78             |
| 4                           | 85.48    | 85.39 | 85.44    | 85.38    | 93.08    | 93.08    | 74.01           | 73.65          |
| 5                           | 85.59    | 85.29 | 86.25    | 86.24    | 92.28    | 92.2     | 74.73           | 74.37          |
| 6                           | 85.59    | 85.59 | 85.7     | 85.6     | 92.6     | 92.51    | 66.79(excluded) | 65.7(excluded) |
| 7                           | 85.3     | 85.22 | 85.59    | 85.55    | 92.35    | 92.09    | 77.26           | 77.26          |
| 8                           | 85.27    | 85.01 | 85.99    | 85.87    | 92.4     | 92.18    | 74.37           | 74.37          |
| Average                     | 85.3675  | 85.2  | 85.80875 | 85.69875 | 92.6325  | 92.51375 | 75.50714        | 75.14571       |
| Best                        | 85.59    | 85.59 | 86.25    | 86.24    | 93.08    | 93.08    | 78              | 78             |
| Worst                       | 85.02    | 84.83 | 85.44    | 85.37    | 92.28    | 92.09    | 72.92           | 72.56          |
| Epochs                      | 2        |       | 2        |          | 2        |          | 4               |                |
| Learning Rate               | 1.00E-05 |       | 2.00E-05 |          | 2.00E-05 |          | 1.00E-05        |                |
| Evaluate every (steps)        | 1000     |       | 1000     |          | 300      |          | 100             |                |
| Total number of evaluations | 26       |       | 26       |          | 23       |          | 5               |                |
| Batch size                  | 32       |       | 32       |          | 32       |          | 32              |                |

* Explanation of the table (CoLA as an example)

The learning rate is set to 5e-6 and the training epoch number is set to 10. The batch size is set to 32. 
Checkpoints are saved and evaluated every 100 steps and 28 checkpoints (28 roughly equals to 10\*8551/(32\*100)+1) are evaluated in total. Eight runs are conducted with randomized random_seed. 
For the first run, 63.42 refers to the best accuracy on the dev set among 28 evaluated checkpoints and
 62.13 refers to the accuracy of the last step checkpoint. We provide the complete results as the finetune process is unstable, where the results vary a lot even with different random seeds.

## LASTEST NEWS

