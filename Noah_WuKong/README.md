# WukongOpenSource

Code for paper _“Wukong: 100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework”_ ([arXiv:2202.06767](https://arxiv.org/abs/2202.06767))

## Code structure

```
.
├── configs/...                     # contains configs for model loading
├── data
│   ├── __init__.py
│   ├── datasets.py                 # definition of datasets, e.g., ImageNet
│   ├── res
│   │   ├── classnames.json         # definition of classification names
│   │   └── prompts.txt             # prompts for ensemble
│   └── tokenizer
│       ├── __init__.py
│       ├── res
│       │   └── vocab.txt           # vocabulary file for tokenization
│       ├── simple_tokenizer.py     # implementation of Chinese tokenization
│       └── utils.py
├── main.py                         # main script for model evaluation
├── model
│   ├── __init__.py
│   ├── builder.py
│   ├── language
│   │   ├── __init__.py
│   │   └── transformer.py          # module of text encoder
│   ├── modules.py                  # some other modules
│   ├── utils.py
│   ├── vision
│   │   ├── __init__.py
│   │   ├── swin_transformer.py     # module of vision encoder [swin-transformer]
│   │   └── vision_transformer.py   # module of vision encoder [vit]
│   └── wukong.py                   # model backbone
├── README.md
├── requirements.txt
└── utils.py
```

## Download models

Benchmark of our pretrained multi-modality models can be found in [Noah-Wukong Benchmark](https://wukong-dataset.github.io/wukong-dataset/benchmark.html)

## Evaluate on ImageNet

Below is an example for evaluating using Wukong_ViT-L model.

```shell
python main.py \
  --config="configs/wukong_vit_l/wukong_vit_l.py" \
  --checkpoint="/cache/ckpt/wukong_vit_l.ckpt" \
  --data_dir="/cache/data/ILSVRC/"
```

## Reference

Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Minzhe Niu, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, Chunjing Xu, Hang Xu.
[Wukong: 100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework](https://arxiv.org/abs/2202.06767).
```
@misc{gu2022wukong,
  title = {Wukong: 100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework},
  author = {Gu, Jiaxi and Meng, Xiaojun and Lu, Guansong and Hou, Lu and Niu, Minzhe and Liang, Xiaodan and Yao, Lewei and Huang, Runhui and Zhang, Wei and Jiang, Xin and Xu, Chunjing and Xu, Hang},
  url = {https://arxiv.org/abs/2202.06767},
  year = {2022}
}
```