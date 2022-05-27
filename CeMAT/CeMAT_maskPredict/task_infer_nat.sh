DATA_PATH=
task=translation_self_from_pt
SRC=
TGT=
langs='ar-en,be-en,bg-en,de-en,el-en,en-af,en-cs,en-es,en-fr,en-gu,en-he,en-ja,en-kk,en-lt,en-mt,en-ro,en-ru,en-tr,en-zh,eo-en,et-en,fi-en,hi-en,it-en,ka-en,ko-en,lv-en,mn-en,ms-en,my-en,sr-en,vi-en'
SAVE_DIR=
PRETRAIN=

python generate_cmlm.py  ${DATA_PATH} --fp16 \
    --user-dir CeMAT_maskPredict \
    --path ${PRETRAIN} \
    --task ${task} --decoding-strategy mask_predict \
    --source-lang ${SRC} --target-lang ${TGT} --langs ${langs} --add-lang-token --share-dict \
    --gen-subset test \
    --max-sentences 20 --decoding-iterations 10 --remove-bpe | tee infer.txt

grep ^H  infer.txt \
| sed 's/^H\-//' \
| sort -V \
| cut -f 2 \
| sed 's/\['$TGT'\] //g' \
| sed 's/\['$TGT'\]//g' \
> infer.sys

grep ^T-  infer.txt \
| sed 's/^T\-//' \
| sort -V \
| cut -f 2 \
| sed 's/\['$TGT'\] //g' \
| sed 's/\['$TGT'\]//g' \
> infer.ref
  
sacrebleu --tokenize 'none' -w 2 infer.ref < infer.sys