# Pretrained Language Model

This repository provides the latest pretrained language models and its related optimization techniques developed by Huawei Noah's Ark Lab.

## Directory structure
* [PanGu-α](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α) is a Large-scale autoregressive pretrained Chinese language model with up to 200B parameter. The models are developed under the [MindSpore](https://www.mindspore.cn/en) and trained on a cluster of [Ascend](https://e.huawei.com/en/products/servers/ascend) 910 AI processors.
* [NEZHA-TensorFlow](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow) is a pretrained Chinese language model which achieves the state-of-the-art performances on several Chinese NLP tasks developed under TensorFlow.
* [NEZHA-PyTorch](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch) is the PyTorch version of NEZHA.
* [NEZHA-Gen-TensorFlow](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-Gen-TensorFlow) provides two GPT models. One is Yuefu (乐府), a Chinese Classical Poetry generation model, the other is a common Chinese GPT model.
* [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) is a compressed BERT model which achieves 7.5x smaller and 9.4x faster on inference.
* [TinyBERT-MindSpore](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT-MindSpore) is a MindSpore version of TinyBERT.
* [DynaBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT) is a dynamic BERT model with adaptive width and depth.
* [BBPE](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BBPE) provides a byte-level vocabulary building tool and its correspoinding tokenizer.
* [PMLM](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PMLM) is a probabilistically masked language model. Trained without the complex two-stream self-attention, PMLM can be treated as a simple approximation of XLNet.
* [TernaryBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TernaryBERT) is a weights ternarization method for BERT model developed under PyTorch.
* [TernaryBERT-MindSpore](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TernaryBERT-MindSpore) is the MindSpore version of TernaryBERT.
* [HyperText](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/HyperText) is an efficient text classification model based on hyperbolic geometry theories.
* [BinaryBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/BinaryBERT) is a weights binarization method using ternary weight splitting for BERT model, developed under PyTorch.
* [AutoTinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/AutoTinyBERT) provides a model zoo that can meet different latency requirements.
* [PanGu-Bot](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-Bot) is a Chinese pre-trained open-domain dialog model build based on the GPU implementation of [PanGu-α](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/PanGu-α).
* [CeMAT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/CeMAT) is a universal sequence-to-sequence multi-lingual pre-training language model for both autoregressive and non-autoregressive neural machine translation tasks.
