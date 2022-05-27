# vocab file path.
vocab_path=
# bilignual(multilignual) word translation dict path
wordTrans_path=
# data path(BPE format)
data_path=
prefix=
langs=
# output path
out_path=

python extract_aligned_pairs.py --vocab-path $vocab_path --trans-path $wordTrans_path --data-path $data_path --output-path $out_path --prefix $prefix --langs $langs --add-mask --merge