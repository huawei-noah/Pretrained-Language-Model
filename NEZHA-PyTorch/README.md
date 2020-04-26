# NEZHA PyTorch version
We only provide fine-tuning codes for sentence classification task in this repository. For MRC and sequential labelling task, please see [CLUE](https://github.com/CLUEbenchmark/CLUE).

### requirements

- pytorch==1.1.0
- python==3.5

### download NEZHA-pytorch models

1. Download pretrained NEZHA-base models from [Google Driver](https://drive.google.com/file/d/1HmwMG2ldojJRgMVN0ZhxqOukhuOBOKUb/view?usp=sharing) or [BaiduYunPan](https://pan.baidu.com/s/1xfYy0U2tJb3w3lpJB00H3Q), password: s2fz .
2. Download pretrained NEZHA-large models from [Google Driver](https://drive.google.com/file/d/1EtahNvdjEpugm8juFuPIN_Fs2skFmeMU/view?usp=sharing) or [BaiduYunPan](https://pan.baidu.com/s/1vc1gkzCZ20bLnUI_r2hn6w), password: 9yfy .
3. Put pretrained models in pytorch_nezha/pretrained_models/.

### Run fine-tuning task
```shell
sh run_classifier.sh
```