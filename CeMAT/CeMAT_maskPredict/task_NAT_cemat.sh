DATA_PATH=
task=translation_self_from_pt
SRC=
TGT=
langs='ar-en,be-en,bg-en,de-en,el-en,en-af,en-cs,en-es,en-fr,en-gu,en-he,en-ja,en-kk,en-lt,en-mt,en-ro,en-ru,en-tr,en-zh,eo-en,et-en,fi-en,hi-en,it-en,ka-en,ko-en,lv-en,mn-en,ms-en,my-en,sr-en,vi-en'
ARCH=bert_transformer_seq2seq_big
freq=8
patience=80
valid_subset='valid'
SAVE_DIR=
PRETRAIN=

python train.py ${DATA_PATH} --fp16 \
  --user-dir CeMAT_maskPredict \
  --encoder-normalize-before --decoder-normalize-before \
  --encoder-learned-pos --decoder-learned-pos \
  --task ${task} --from-pt \
  --source-lang ${SRC} --target-lang ${TGT} --langs ${langs} --add-lang-token --share-dict \
  --arch ${ARCH}  --share-all-embeddings \
  --criterion label_smoothed_length_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 \
  --lr 0.0005 --warmup-init-lr 1e-7 --min-lr 1e-9 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --dropout 0.3 --weight-decay 0.01 \
  --max-tokens 4096 --update-freq  ${freq} \
  --max-source-positions 10000 --max-target-positions 10000 --max-update 300000 --seed 0 \
  --restore-file ${PRETRAIN} --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 \
  --log-format simple --log-interval 100 \
  --ddp-backend no_c10d \
  --save-dir ${SAVE_DIR} --patience ${patience} --num-workers 4 \
  --distributed-no-spawn \
  --valid-subset ${valid_subset}