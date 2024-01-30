import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from transformers import BertConfig, BertForSequenceClassification

from transformers.data.processors.multiemo import MultiemoProcessor

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models')
DYNABERT_FT_MODELS_FOLDER = os.path.join(MODELS_FOLDER, 'dynabert-finetuned')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument('--depth_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument('--width_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")

    args = parser.parse_args()
    task_name = args.task_name

    models_subdirectories = get_immediate_subdirectories(DYNABERT_FT_MODELS_FOLDER)
    print(DYNABERT_FT_MODELS_FOLDER)
    print(models_subdirectories)

    data = list()
    for subdirectory in models_subdirectories:
        data_dict = gather_results(subdirectory, task_name, args.depth_mult, args.width_mult)
        data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results-dynabert-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(model_dir: str, task_name: str, depth_mult: float, width_mult: float) -> Dict[str, Any]:
    task_subfolder = os.path.basename(model_dir)
    eval_output_dir = os.path.join(
        model_dir, 'bert_' + str(width_mult) + '_' + str(depth_mult) + '_eval')

    with open(os.path.join(model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(eval_output_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    with open(os.path.join(MODELS_FOLDER, 'dynabertw', task_subfolder, 'training_params.json')) as json_file:
        dynabertw_training_data_dict = json.load(json_file)
        data['training_time'] = data['training_time'] + dynabertw_training_data_dict['training_time']

    with open(os.path.join(MODELS_FOLDER, 'dynabert', task_subfolder, 'training_params.json')) as json_file:
        dynabert_training_data_dict = json.load(json_file)
        data['training_time'] = data['training_time'] + dynabert_training_data_dict['training_time']

    model_size = os.path.getsize(os.path.join(model_dir, 'pytorch_model.bin'))
    data['model_size'] = model_size

    if 'multiemo' not in task_name:
        raise ValueError("Task not found: %s" % task_name)

    _, lang, domain, kind = task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # LOADING THE BEST MODEL
    config = BertConfig.from_pretrained(model_dir, num_labels=num_labels, finetuning_task=task_name)
    model = BertForSequenceClassification.from_pretrained(model_dir, config=config)
    model.apply(lambda m: setattr(m, 'depth_mult', float(depth_mult)))
    model.apply(lambda m: setattr(m, 'width_mult', float(width_mult)))

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    data['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    data['parameters'] = parameters_num
    data['name'] = os.path.basename(model_dir)
    data['model_name'] = 'Dynabert_d_' + str(depth_mult) + '_w_' + str(width_mult)
    print(data)
    return data


if __name__ == '__main__':
    main()
