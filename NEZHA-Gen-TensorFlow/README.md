
NEZHA-Gen-TensorFlow
=============
We provide two GPT models pretrained by Huawei Noah's Ark Lab. One is Yuefu (乐府), a Chinese Classical Poetry generation model. The other is a Chinese GPT model pretrained with Chinese wikipedia and news corpus


Release Notes
=============
First version: 2020/07/22

Yuefu updated: 2020/09/24

Environment
============
The scripts are tested sucessfully with Tensorflow 1.13 and Python 3.6. 

The python package ``fire`` is required. You may need to install the ``fire`` package with the command:

```
pip3 install fire
```

Usage of Yuefu (乐府)
====================

Step 1: Download the folder named ``models_yuefu`` from the link below and move the folder to the same directory with the scripts. Rename the folder to ``models``.

Step 2: Run the script ``poetry.py`` with the command to see a demo output:

```
python3 poetry.py
```

The opensourced Yuefu is only for academic research.
Any business application should refer to [Huawei Cloud API](https://support.huaweicloud.com/api-nlp/nlp_03_0070.html).


Usage of GPT
====================

Step 1: Download the folder named ``models_gpt`` from the link below and move the folder to the same directory with the scripts. Rename the folder to ``models``.

Step 2: Run the script ``interactive_conditional_generation.py`` with the command:

```
python3 interactive_conditional_generation.py
```

Step 3: Type in Chinese characters as the initial words and press ENTER to start generating sentences.

Model download 
===========================
* Yuefu
    * [Google Drive](https://drive.google.com/drive/folders/1B5-jxUlzhoKwFVMQ-nkqqbmJQgr1lRAp?usp=sharing) 
    * [Baidu Netdisk](https://pan.baidu.com/s/1me6_BGYHbWFdTi80vRQ2Lg)(code: ytim)

* Chinese GPT
    * [Google Drive](https://drive.google.com/drive/folders/1i4f_8LhaVDNjnGlLXNJ0rNgBP0E4L6V0?usp=sharing) 
    * [Baidu Netdisk](https://pan.baidu.com/s/1Bgle8TpcxHyuUz_jAXOBWw)(code:rb5m)

