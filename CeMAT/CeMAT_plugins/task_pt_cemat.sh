DATA_PATH=
task=cemat_pretraining
langs='ar-en,be-en,bg-en,de-en,el-en,en-af,en-cs,en-es,en-fr,en-gu,en-he,en-ja,en-kk,en-lt,en-mt,en-ro,en-ru,en-tr,en-zh,eo-en,et-en,fi-en,hi-en,it-en,ka-en,ko-en,lv-en,mn-en,ms-en,my-en,sr-en,vi-en'
word_trans='word_trans2id.dict'
ARCH=cemat_transformer_big
# with 32 GPUS.
freq=8
patience=10
valid_subset='valid'
SAVE_DIR=

fairseq-train ${DATA_PATH} --fp16 \
  --user-dir CeMAT_plugins \
  --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
  --task ${task} \
  --langs ${langs} --add-lang-token --share-dict --shuffle-lang-pair --multilang-sampling-alpha 0.7 \
  --trans-dict ${word_trans} \
  --arch ${ARCH} --bi_self_att --plus-encoder-loss --encoder-loss-lambda 0.3 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --criterion label_smoothed_cross_entropy_with_maskdecode --label-smoothing 0.1 \
  --lr 0.0005 --lr-scheduler polynomial_decay --warmup-updates 10000 --total-num-update 1200000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0001 \
  --max-tokens 4096 --update-freq ${freq} --seed 222 \
  --log-format simple --skip-invalid-size-inputs-valid-test \
  --ddp-backend c10d \
  --keep-interval-updates 20 --log-interval 10 \
  --validate-interval 1 \
  --save-dir ${SAVE_DIR} --num-workers 4 --patience ${patience} \
  --valid-subset ${valid_subset} \
