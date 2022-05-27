SRC=
TGT=
DATA=
OUTPATH=
DEST=
# vocab,model path.
MODEL=
mkdir -p ${OUTPATH}
mkdir -p ${DEST}

N_THREADS=8
# if need
pip install jieba

FASTBPE_DIR=
FASTBPE=
BPEROOT=

#moses decoder path
MOSES=
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
CLEAN=$MOSES/scripts/training/clean-corpus-n.perl
NORMALIZE_ROMANIAN=$MOSES/scripts/tokenizer/ro/normalise-romanian.py
REMOVE_DIACRITICS=$MOSES/scripts/tokenizer/ro/remove-diacritics.py
JA_SCRIPT=$MOSES/scripts/tokenizer/ja/kytea.py
JA_MODEL=$MOSES/scripts/tokenizer/ja/ja-0.4.7-1.mod

# BPE / vocab files
BPE_CODES=$MODEL/codes
FULL_VOCAB=$MODEL/vocab

for split in "train" "valid" "test";
do
    for lang in $SRC $TGT;
    do
        Data_TRAIN=$DATA/$split.$lang
        Data_TRAIN_TOK=$OUTPATH/$split.tok.$lang
        Data_TRAIN_BPE=$OUTPATH/$split.spm.$lang
        echo $Data_TRAIN "TOKENIZER:====>>" $Data_TRAIN_TOK
        if [ "$lang" == "ro" ]; then 
            cat $Data_TRAIN | perl $NORM_PUNC -l $lang | perl $REM_NON_PRINT_CHAR | perl $NORMALIZE_ROMANIAN | perl $REMOVE_DIACRITICS | perl $TOKENIZER -l $lang -a -threads $N_THREADS > $Data_TRAIN_TOK
        elif [ "$lang" == "ja" ]; then 
            cat $Data_TRAIN | perl $NORM_PUNC -l $lang | perl $REM_NON_PRINT_CHAR | python ${JA_SCRIPT} -m ${JA_MODEL}             > $Data_TRAIN_TOK
        elif [ "$lang" == "zh" ]; then 
            cat $Data_TRAIN | perl $NORM_PUNC -l $lang | perl $REM_NON_PRINT_CHAR | python -m jieba -d                             > $Data_TRAIN_TOK
        else
            cat $Data_TRAIN | perl $NORM_PUNC -l $lang | perl $REM_NON_PRINT_CHAR | perl $TOKENIZER -l $lang -a -threads $N_THREADS > $Data_TRAIN_TOK
        fi

        echo $Data_TRAIN_TOK "====>>" $Data_TRAIN_BPE
        python $BPEROOT/apply_bpe.py -c $BPE_CODES < $Data_TRAIN_TOK > $Data_TRAIN_BPE
    done
    if [ "$split" == "train" ]; then 
        echo "clean by ratio."
        perl $CLEAN -ratio 1.5 $OUTPATH/$split.spm $SRC $TGT $OUTPATH/$split.spm.clean 1 250
    fi
done

#Binarize the dataset
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${OUTPATH}/train.spm.clean \
  --validpref ${OUTPATH}/valid.spm \
  --testpref ${OUTPATH}/test.spm \
  --destdir ${DEST}/ \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $FULL_VOCAB \
  --tgtdict $FULL_VOCAB \
  --workers 70
