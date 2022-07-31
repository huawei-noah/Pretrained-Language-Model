# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import math
import time
import logging
import datetime
import argparse
from io import BytesIO
from multiprocessing import Pool

import pandas as pd
from PIL import Image
from mindspore.mindrecord import FileWriter

from ..tools import set_tokenizer_lang, tokenize
set_tokenizer_lang('zh')
logger = logging.getLogger(__name__)


def get_file_id(csv_file_name):
    return int(os.path.splitext(csv_file_name)[0].split('_')[-1])


def process_one_data(data_info):
    img_file_path, text = data_info
    sample_dict = dict()
    if not os.path.isfile(img_file_path):
        return None
    img_buf = Image.open(img_file_path)
    img_bytes_io = BytesIO()
    img_buf.save(img_bytes_io, 'JPEG')

    if isinstance(text, str):
        text = [text]
    token_data = tokenize(text)

    sample_dict['file_name'] = img_file_path
    sample_dict['token'] = token_data
    sample_dict['image'] = img_bytes_io.getvalue()
    return sample_dict


def convert_to_mindrecord(data_info_list, worker_num, file_writer):
    with Pool(worker_num) as pool:
        result = pool.map(process_one_data, data_info_list)
    data_record = []
    for item in result:
        if item is not None:
            data_record.append(item)
    if data_record:
        file_writer.write_raw_data(data_record)


def process_data(csv_dir, img_root, data_record_dir, shard_num, worker_num, block_size):
    data_record_path = os.path.join(data_record_dir, 'wukong.mindrecord')
    writer = FileWriter(file_name=data_record_path, shard_num=shard_num)
    data_schema = {"file_name": {"type": "string"},
                   "image": {"type": "bytes"},
                   "token": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, 'wukong 100m schema')
    writer.add_index(["file_name"])

    start_time = time.time()
    csv_file_list = list(os.listdir(csv_dir))
    for csv_file_name in csv_file_list:
        file_id = get_file_id(csv_file_name)
        img_dir = os.path.join(img_root, '{:03d}'.format(file_id))
        csv_data = pd.read_csv(os.path.join(csv_dir, csv_file_name))
        csv_len = len(csv_data)
        block_cnt = math.ceil(csv_len / block_size)
        block_id = 0
        data_info_list = []
        for img_id, text in enumerate(csv_data['caption']):
            img_file_path = os.path.join(img_dir, '{:05d}.jpg'.format(img_id))
            data_info_list.append((img_file_path, text))
            if len(data_info_list) == block_size:
                convert_to_mindrecord(data_info_list, worker_num, writer)
                block_id += 1
                cur_time = time.time()
                ave_speed = (cur_time - start_time) / block_id
                eta_seconds = (block_cnt - block_id) * ave_speed
                eta = str(datetime.timedelta(seconds=eta_seconds))
                logger.info('processing %s, block: %d/%d, ave speed: %.2f s/block, eta: %s',
                            csv_file_name, block_id, block_cnt, ave_speed, eta)
                data_info_list = []
        if data_info_list:
            convert_to_mindrecord(data_info_list, worker_num, writer)
    writer.commit()


def main():
    parser = argparse.ArgumentParser(description='wukong dataset converter')
    parser.add_argument('--csv_dir', help='dir path for csv file', required=True)
    parser.add_argument('--img_dir', help='dir path for image file', required=True)
    parser.add_argument('--data_record_dir', help='mindrecord generate dir', required=True)
    parser.add_argument('--shard_num', help='mindrecord shard number', type=int, default=10)
    parser.add_argument('--worker_num', help='parallel worker number to process data', type=int, default=4)
    parser.add_argument('--block_size', help='block size for each mindrecord write', type=int, default=2000)

    args = parser.parse_args()
    csv_dir = args.csv_dir
    img_dir = args.img_dir
    data_record_dir = args.data_record_dir
    shard_num = args.shard_num
    worker_num = args.worker_num
    block_size = args.block_size

    if not os.path.isdir(csv_dir):
        logger.error('csv dir %s not exists', csv_dir)
        raise FileNotFoundError
    if not os.path.isdir(img_dir):
        logger.error('img dir %s not exists', img_dir)
        raise FileNotFoundError
    if not os.path.isdir(data_record_dir):
        logger.error('data record dir %s not exists', data_record_dir)
        raise FileNotFoundError
    process_data(csv_dir, img_dir, data_record_dir, shard_num, worker_num, block_size)


if __name__ == '__main__':
    main()
