import TextSharding

# Segmentation is here because all datasets look the same in one article/book/whatever per line format, and
# it seemed unnecessarily complicated to add an additional preprocessing step to call just for this.
# Different languages (e.g., Chinese simplified/traditional) may require translation and
# other packages to be called from here -- just add a conditional branch for those extra steps
segmenter = TextSharding.NLTKSegmenter()
sharding = TextSharding.Sharding(['/home/ma-user/work/Old_BERT/data/origin/wiki_sliced/wiki_00', '/home/ma-user/work/Old_BERT/data/origin/wiki_sliced/wiki_01'], '/home/ma-user/work/Old_BERT/data/origin/wiki_sliced', 256, 256, 0.1)

sharding.load_articles()
sharding.segment_articles_into_sentences(segmenter)
sharding.distribute_articles_over_shards()
sharding.write_shards_to_disk()
