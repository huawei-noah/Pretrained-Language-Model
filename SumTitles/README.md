This repository contains code for paper V. Malykh, K. Chernis, E. Artemova, I. Pionkovskaya. "SumTitles: a Summarization Corpus with Low Extractivity". COLING 2020.

To collect a Subtitles part of the corpus please download English monolingual subcorpus of [OpenSubtitles-v2018](http://opus.nlpl.eu/OpenSubtitles-v2018.php).
Please unpack it to the default folder `./en/OpenSubtitles/xml/en` or another place by your choice. Then you need to run 
`cd opensubtitles_corpus; python3 opensubtitles_processing.py` to process and align the subtitles.

To collect a Scripts part of the corpus you need to download movie scripts first.
To download the scripts please use `cd kvtoraman_scraper; sh scrap_films.sh`

After that you need to make an alignment.
To make alignment for the scripts, please use command `sh ./parse.sh`.