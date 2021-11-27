import os
import zipfile

import requests
from tqdm.auto import tqdm

# url = 'https://clarin-pl.eu/dspace/bitstream/handle/11321/798/multiemo.zip?sequence=2&isAllowed=y'
url = 'https://clarin-pl.eu/dspace/handle/11321/798/allzip'


def main(data_dir):
    output_zip = os.path.join(
        data_dir,
        'MultiEmo_ Multilingual, Multilevel, Multidomain Sentiment Analysis Corpus of Consumer Reviews.zip')

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(output_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    progress_bar.update(len(chunk))
                    f.write(chunk)

        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(output_zip)
    os.remove(os.path.join(data_dir, 'multiemo.7z'))

    data_output_zip = os.path.join(data_dir, 'multiemo.zip')
    with zipfile.ZipFile(data_output_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(data_output_zip)
    os.remove(os.path.join(data_dir, 'README.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='glue_data')
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)

    main(data_dir=args.data_dir)
