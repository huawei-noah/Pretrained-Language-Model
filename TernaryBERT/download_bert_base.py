import os
import requests
import tarfile

url = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz'

output_path = os.path.join('data', 'models')
os.makedirs(output_path, exist_ok=True)

output_tar = os.path.join(output_path, 'bert-base-uncased.tar.gz')
model_folder = os.path.join(output_path, 'bert-base-uncased')

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_tar, 'wb') as f:
        f.write(response.raw.read())

with tarfile.open(name=output_tar, mode="r|gz") as tar_ref:
    tar_ref.extractall(model_folder)

os.rename(os.path.join(model_folder, 'bert_config.json'), os.path.join(model_folder, 'config.json'))

os.remove(output_tar)

url_vocab = 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'
r = requests.get(url_vocab)

with open(os.path.join(model_folder, 'vocab.txt'), 'wb') as f:
    f.write(r.content)

print('Completed!')
