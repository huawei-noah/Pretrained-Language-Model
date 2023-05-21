python /home/ma-user/work/Old_BERT/create_pretraining_data.py \
	--input_file=/cache/data/book/book_corpus_2.txt  \
	--output_file=/cache/data/book/book_corpus_2.hdf5 \
	--vocab_file=/home/ma-user/work/Old_BERT/bert-large-uncased-vocab.txt \
	--bert_model=bert-large-uncased \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--dupe_factor=5 \
	--masked_lm_prob=0.15 
