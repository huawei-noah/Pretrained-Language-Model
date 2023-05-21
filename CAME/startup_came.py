"""
Startup script to run on the cloud
"""

import moxing
import os
import argparse
import logging

# install libraries

os.environ["NUMBA_NUM_THREADS"] = '1'

os.system('pip install setuptools==59.0.1')
os.system('pip install torchmetrics==0.7.1')
print('Install torch finished...')

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
parser.add_argument('--data_url', type=str)
parser.add_argument('--train_url', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--learning_rate', type=float)
parser.add_argument('--max_steps', type=int)
parser.add_argument('--gradient_accumulation_steps', type=int)


args, unparsed = parser.parse_known_args()
print(args, unparsed)

# download data

moxing.file.copy_parallel('/pretraining_data/train_data','/cache/data/train_data' )
moxing.file.copy_parallel('/pretraining_data/dev_all_data','/cache/data/dev_data' )

# run program
import os

os.system('bash /home/ma-user/work/BERT/run_came_pretraining.sh')
