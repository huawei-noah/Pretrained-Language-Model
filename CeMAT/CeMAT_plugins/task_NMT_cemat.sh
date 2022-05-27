DATA_PATH=
task=translation_from_pretrained_cemat
SRC=
TGT=
langs='ar-en,be-en,bg-en,de-en,el-en,en-af,en-cs,en-es,en-fr,en-gu,en-he,en-ja,en-kk,en-lt,en-mt,en-ro,en-ru,en-tr,en-zh,eo-en,et-en,fi-en,hi-en,it-en,ka-en,ko-en,lv-en,mn-en,ms-en,my-en,sr-en,vi-en'
ARCH=cemat_transformer_big
freq=16
patience=35
valid_subset='valid'
SAVE_DIR=
PRETRAIN=

fairseq-train ${DATA_PATH} --fp16 \
  --user-dir CeMAT_plugins \
  --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
  --task ${task} \
  --source-lang ${SRC} --target-lang ${TGT} --langs ${langs} --add-lang-token --share-dict \
  --arch ${ARCH} \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --attention-dropout 0.1 \
  --max-tokens 4096 --update-freq ${freq} --seed 222 \
  --log-format simple --skip-invalid-size-inputs-valid-test \
  --keep-interval-updates 20 --log-interval 10 \
  --validate-interval 1 \
  --restore-file ${PRETRAIN} --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --eval-bleu-print-samples \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --ddp-backend c10d \
  --save-dir ${SAVE_DIR} --num-workers 4 --patience ${patience} \
  --valid-subset ${valid_subset}