## CeMAT

This repository contains code for the following paper:

>Pengfei Li, Liangyou Li, Meng Zhang, Minghao Wu, Qun Liu.  **Universal Conditional Masked Language Pre-training for Neural Machine Translation**. In ACL2022 main conference paper.

This is a reimplementation based on [fairseq](https://github.com/facebookresearch/fairseq) and [Mask-Predict](https://github.com/facebookresearch/Mask-Predict) (for Non-autoregressive neural machine translation ) which should reproduce reasonable results. 

### Requirements

- Python >= 3.7
- Pytorch >= 1.7.0
- Fairseq 1.0.0a0
- sacrebleu==1.5.1
- fastBPE (for BPE codes)
- Moses (for tokenization)
- Apex (for fp16 training)
- kytea(for Japanese tokenization)
- jieba(for Chinese tokenization)

The pipeline contains two steps: Pre-training and fine-tuning.

* [Pre-training](#1)
 * [Fine-tuning(Autoregressive NMT)](#2.1)
 * [Fine-tuning(Non-autoregressive NMT)](#2.2)

<h2 id="1">Pre-training</h2>

You can simply ignore the current step if you directly download the pre-trained model and vocab we provide.

#### 1. Prepare Data

We use the bilingual and monolingual corpus for pre-training. where we use bilingual data in 32 languages provided by [mRASP](https://github.com/linzehui/mRASP), and the monolingual corpus can be downloaded from [news-crawl](https://data.statmt.org/news-crawl/). 

#### 2. Tokenization，Learn and apply BPE

You can simply ignore the current step if you directly download the data and vocab we provide.

Special tokenization for Romanian, Chinese and Japanese, we directly use the Vocab and BPE Code provided by mRASP. 

```bash
# bilingual
bash ${PROJECT_ROOT}/cemat_scripts/process/preprocess_Para.sh
# monolingual
bash ${PROJECT_ROOT}/cemat_scripts/process/preprocess_Mono.sh
```

if you want to use you own code, should  learn your own BPE code before applying BPE, and get vocab file base on subword data.

```bash
FASTBPE=fastBPE/fast
# merge operation nums.
CODES=35000
SRC_TRAIN_TOK=
TGT_TRAIN_TOK=
#output
BPE_CODES=
VOCAB=
SRC_TRAIN_BPE=
TGT_TRAIN_BPE=

$FASTBPE learnbpe $CODES ${SRC_TRAIN_TOK} ${TGT_TRAIN_TOK} > $BPE_CODES
$FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TRAIN_TOK $BPE_CODES
$FASTBPE applybpe $TGT_TRAIN_BPE $TGT_TRAIN_TOK $BPE_CODES
$FASTBPE getvocab $SRC_TRAIN_BPE $TGT_TRAIN_BPE > $VOCAB
```

#### 3. Binarize the dataset

For bilignual.

```bash
SRC=
TGT=
# your bpe corpus path
OUTPATH=
#OUTPUT
DEST=

fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${OUTPATH}/train.${SRC}-${TGT}.spm.clean \
  --validpref ${OUTPATH}/valid.spm \
  --testpref ${OUTPATH}/test.spm \
  --destdir ${DEST}/ \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $VOCAB \
  --tgtdict $VOCAB \
  --workers 70
```

For monolingual.

```bash
lang=
# your bpe corpus path
OUTPATH=
#OUTPUT
DEST=

fairseq-preprocess \
  --only-source \
  --source-lang ${lang} \
  --trainpref ${OUTPUT}/train.spm \
  --validpref ${OUTPUT}/valid.spm \
  --destdir ${DEST}/ \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $VOCAB \
  --tgtdict $VOCAB \
  --workers 70
mv ${DEST}/train.$lang-None.$lang.bin ${DEST}/train.$lang-$lang.$lang.bin
mv ${DEST}/train.$lang-None.$lang.idx ${DEST}/train.$lang-$lang.$lang.idx
mv ${DEST}/valid.$lang-None.$lang.bin ${DEST}/valid.$lang-$lang.$lang.bin
mv ${DEST}/valid.$lang-None.$lang.idx ${DEST}/valid.$lang-$lang.$lang.idx
```

#### 4. Create Aligned word pairs between source and target.

You can simply ignore the current step if you directly download the data(train.merge.json, valid.merge.json) we provide.

```bash
# vocab file path.
vocab_path=
# bilignual(multilignual) word translation dict path(word_trans2id.dict)
wordTrans_path=
# data path(BPE format)
data_path=
prefix=
# 'zh-en'
langs=
# output path
out_path=

python extract_aligned_pairs.py --vocab-path $vocab_path --trans-path $wordTrans_path --data-path $data_path --output-path $out_path --prefix $prefix --langs $langs --add-mask --merge
```

#### 5. Training

After above preprocesses, all training data is ready. Please put all the dictionaries (including word_trans2id.dict, train.merge.json, valid.merge.json,dict.txt) into the **./Dicts** directory, which is in the same place as the binarize data.

```bash
bash ${PROJECT_ROOT}/CeMAT_plugins/task_pt_cemat.sh
```

You can modify the configs to choose the model architecture or dataset used.

<h2 id="2.1">Fine-tuning(Autoregressive NMT)</h2>

#### 1. Preprocess and binarize data

```bash
# bilingual
bash ${PROJECT_ROOT}/process/preprocess_NMT.sh
```

#### 2. Fine-tune on specific language pairs

```bash
bash ${PROJECT_ROOT}/CeMAT_plugins/task_NMT_cemat.sh
```

#### 3. Inference

```bash
bash ${PROJECT_ROOT}/CeMAT_plugins/task_infer_nmt.sh
```

<h2 id="2.2">Fine-tuning(Non-autoregressive NMT)</h2>

#### 1. Preprocess and binarize data

```bash
# bilingual
bash ${PROJECT_ROOT}/process/preprocess_NMT.sh
```

#### 2. Fine-tune on specific language pairs

```bash
bash ${PROJECT_ROOT}/CeMAT_maskPredict/task_NAT_cemat.sh
```

#### 3. Inference

```bash
bash ${PROJECT_ROOT}/CeMAT_maskPredict/task_infer_nat.sh
```

## License

CeMAT is MIT License, however, there is one exception that needs to be noted that the license of CeMAT_maskPredict(for non-autoregressive NMT) is CC-BY-NC 4.0, which is limited by the license of [MASK-PREDICT](https://github.com/facebookresearch/Mask-Predict). The license applies to the pre-trained models as well.

## Citation

If you find this repo helpful for your research, please cite:

```
@inproceedings{CeMAT,
  author    = {Li, Pengfei and Li, Liangyou and Zhang, Meng and Wu, Minghao and Liu, Qun},
  title     = {Universal Conditional Masked Language Pre-training for Neural Machine Translation},
  booktitle   = {Annual Meeting of the Association for Computational Linguistics},
  year      = {2022}
}
```
