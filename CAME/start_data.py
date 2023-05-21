"""
Startup script to run on the cloud
"""

import moxing
import os
import argparse
import logging

# install libraries

os.environ["NUMBA_NUM_THREADS"] = '1'

# os.system('python -m pip install --upgrade --force pip ')
os.system('pip install setuptools==59.0.1')
os.system('pip install torchmetrics==0.7.1')
# os.system('pip install --upgrade --force torch==1.7')
print('Install torch finished...')

#moxing.file.copy_parallel('s3://bucket-1495/mengxinfan/wheel/', '/cache/wheel')
# os.system('pip install --force /cache/wheel/apex-0.1-cp36-cp36m-linux_x86_64.whl')
# os.system('pip install --force /cache/wheel/mlperf_logging-0.0.0-py3-none-any.whl')
moxing.file.copy_parallel('s3://bucket-3531/luoyang/work/DeepLearningExamples/PyTorch/LanguageModeling/BERT', '/home/ma-user/work/Old_BERT')
# os.system('pip install --force /cache/wheel/apex-0.1-cp36-cp36m-linux_x86_64.whl')
# os.system('pip install --force /cache/wheel/mlperf_logging-0.0.0-py3-none-any.whl')
moxing.file.copy_parallel('s3://bucket-3531/luoyang/work/apex-2020', '/home/ma-user/work/apex-2020')

os.system('pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /home/ma-user/work/apex-2020/')

moxing.file.copy_parallel('s3://bucket-3531/luoyang/work/dllogger', '/home/ma-user/work/dllogger')

os.system('pip install /home/ma-user/work/dllogger')
print('Install apex and dllogger finished...')

os.system('mkdir /cache/results')
print('Result directory created')

os.system('pip install pyarrow==2.0.0')
os.system('pip install tqdm')
os.system('pip install h5py')
os.system('pip install onnxruntime==1.0.0')
os.system('pip install boto3==1.15.0')
os.system('pip install torch-optimizer==0.0.1a16')
os.system('pip install torch-SM3')

logging.info("finish install tqdm and h5py")


try:
    import torch
    print('Import torch success...')
    print('torch version: ', torch.__version__)
    print('cuda status: ', torch.cuda.is_available())
    import apex
    print('Import apex success...')
    import amp_C
    print('Import amp_C success...')
    import apex_C
    print('Import apex_C success...')
except Exception as e:
    print('Some failure...', e)


parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default="")
parser.add_argument('--train_url', type=str, default="")
parser.add_argument('--batch_size', type=int, default=8192)
parser.add_argument('--learning_rate', type=float, default=0.0025)
parser.add_argument('--max_steps', type=int, default=12500)
parser.add_argument('--gradient_accumulation_steps', type=int, default=512)


args, unparsed = parser.parse_known_args()
print(args, unparsed)

# download data

#moxing.file.copy_parallel('s3://bucket-373/xiaozhe/data/data/NLP/bert_en_hdf5_128/', '/home/ma-user/work/Old_BERT/data/pretrain_data')
moxing.file.copy_parallel('s3://bucket-3531/xiaozhe/data/bert_en/origin/book/book_corpus_2','/cache/data/book/book_corpus_2.txt' )

#moxing.file.copy_parallel('s3://bucket-3531/luoyang/pretraining_data/dev_all_data', '/cache/data/dev_data' )

#moxing.file.copy_parallel('s3://bucket-3531/luoyang/lamb_32k_ckpts/ckpt_9988.pt', '/cache/ckpt_9988.pt')
#print('Checkpoint loaded...')
# download checkpoint
# os.system('mkdir /cache/checkpoint /cache/config /cache/results')
# moxing.file.copy_parallel('s3://bucket-1495/mengxinfan/bert/bert/checkpoint/model.ckpt-28252.pt', '/cache/checkpoint/model.ckpt-28252.pt')
# moxing.file.copy_parallel('s3://bucket-1495/mengxinfan/bert/bert/checkpoint/bs64k_32k_ckpt_bert_config.json', '/cache/config/bs64k_32k_ckpt_bert_config.json')

# moxing.file.copy_parallel('s3://gpt-next/zhoupingyi/ckpt/bert_nv_perf/model.ckpt-28252.pt', '/cache/checkpoint/model.ckpt-28252.pt')
# moxing.file.copy_parallel('s3://gpt-next/zhoupingyi/ckpt/bert_nv_perf/bs64k_32k_ckpt_bert_config.json', '/cache/config/bs64k_32k_ckpt_bert_config.json')



# os.system('ls -larth /cache/config')
# os.system('mount')
# os.system('df -h')
# os.system('ls -larth /cache/checkpoint')


# run program
import os

#env_str = " %s=%s %s=%s %s=%s %s=%s" % ('BATCH_SIZE', args.batch_size, 'MAX_STEPS', args.max_steps, 'LEARNING_RATE', args.learning_rate, 'GRADIENT_ACCUMULATION_STEPS', args.gradient_accumulation_steps)


os.system('bash /home/ma-user/work/Old_BERT/create_data.sh')

#os.system("ls -larth /home/ma-user/work/Old_BERT/results")
#import moxing
#moxing.file.copy_parallel('/cache/data/book/book_corpus_3.hdf5', 's3://bucket-3531/luoyang/book/book_corpus_3.hdf5')
#output_url = "s3:/" if not args.train_url.startswith('s3:/') else ""
#moxing.file.copy_parallel('/home/ma-user/work/Old_BERT/results/', 's3://bucket-3531/luoyang/output_adafalamb_4096/')
