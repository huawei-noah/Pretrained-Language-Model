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
import time
import queue
import logging
import datetime
import warnings
import argparse
import threading
from io import BytesIO

import requests
from PIL import Image
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_img_func(q, img_download_root, total_len):
    start_time = time.time()
    while True:
        try:
            url_info = q.get_nowait()
        except queue.Empty:
            break
        file_id, img_id, img_url = url_info
        if (img_id + 1) % 1000 == 0:
            cur_time = time.time()
            time_used = cur_time - start_time
            avg_speed = (img_id + 1) / time_used
            eta_seconds = (total_len - (img_id + 1)) / avg_speed
            eta = str(datetime.timedelta(seconds=eta_seconds))
            logger.info('%d/%d images downloaded, ave speed: %.2f fps, eta: %s',
                        (img_id + 1), total_len, avg_speed, eta)
        img_download_dir = os.path.join(img_download_root, '{:03d}'.format(file_id))
        img_save_path = os.path.join(img_download_dir, '{:05}.jpg'.format(img_id))
        if os.path.exists(img_save_path):
            continue
        try:
            res = requests.get(img_url, stream=True, verify=False, timeout=5)
            if res.status_code == 200:
                buf = BytesIO()
                buf.write(res.content)
                img = Image.open(buf)

                if img.format == 'PNG' and img.mode == 'RGBA':
                    background = Image.new('RGBA', img.size, (255, 255, 255))
                    background.paste(img, img)
                    img = background.convert('RGB')
                elif img.mode == 'P':
                    img = img.convert('RGBA')
                    background = Image.new('RGBA', img.size, (255, 255, 255))
                    background.paste(img, img)
                    img = background.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(img_save_path)
        except requests.RequestException as e:
            logger.exception(e)


def open_threads(target_func, thread_args, thread_num):
    t = []
    for i in range(thread_num):
        list_args = list(thread_args)
        t.append(threading.Thread(target=target_func, args=tuple(list_args), name='child_thread_%d' % i))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()


def main():
    parser = argparse.ArgumentParser(description='downloader for wukong dataset')
    parser.add_argument('--csv_dir', help='dir path for csv file', required=True)
    parser.add_argument('--img_dir', help='path to save images to', required=True)
    parser.add_argument('--start_id', help='start id for csv file', type=int, default=0)
    parser.add_argument('--end_id', help='end id for csv file', type=int, default=-1)
    parser.add_argument('--thread_num', help='download thread count', type=int, default=4)

    args = parser.parse_args()
    csv_dir = args.csv_dir
    img_dir = args.img_dir
    start_id = args.start_id
    end_id = args.end_id
    thread_num = args.thread_num

    if not os.path.isdir(csv_dir):
        logger.error('csv dir %s not exists', csv_dir)
        raise FileNotFoundError
    if not os.path.isdir(img_dir):
        logger.error('img dir %s not exists', img_dir)
        raise FileNotFoundError

    csv_file_list = list(os.listdir(csv_dir))
    csv_file_num = len(csv_file_list)
    if end_id < 0:
        end_id = csv_file_num + end_id + 1
    csv_file_template = 'wukong_100m_{}.csv'

    q = queue.Queue()
    for file_id in range(start_id, end_id):
        csv_filepath = os.path.join(csv_dir, csv_file_template.format(file_id))
        data = pd.read_csv(csv_filepath, lineterminator='\n')
        img_download_dir = os.path.join(img_dir, '{:03d}'.format(file_id))
        if not os.path.isdir(img_download_dir):
            os.mkdir(img_download_dir)
        for img_id, download_url in enumerate(data['url']):
            download_info = (file_id, img_id, download_url)
            q.put(download_info)
            if img_id > 10000:
                break
    total_len = q.qsize()
    open_threads(target_func=fetch_img_func, thread_args=(q, img_dir, total_len), thread_num=thread_num)


if __name__ == '__main__':
    main()
