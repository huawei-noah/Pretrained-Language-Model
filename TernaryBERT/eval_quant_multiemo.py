from __future__ import absolute_import, division, print_function

import argparse
import random
import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import classification_report
from tqdm import tqdm

from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertConfig
from utils_multiemo import *
from utils import dictionary_to_json, result_to_text_file

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels):
    eval_loss = 0
    nb_eval_steps = 0
    all_logits = None

    for batch_ in tqdm(eval_dataloader):
        batch_ = tuple(t.to(device) for t in batch_)
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_
            logits, _, _ = model(input_ids, segment_ids, input_mask)

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if output_mode == "regression":
        all_logits = np.squeeze(all_logits)
    result = compute_metrics(task_name, all_logits, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result, all_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models/tinybert',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--task_name",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument("--weight_bits",
                        default=2,
                        type=int,
                        choices=[2, 8],
                        help="Quantization bits for weight.")
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")

    args = parser.parse_args()
    task_name = args.task_name.lower()
    data_dir = args.data_dir

    model_dir = os.path.join(args.model_dir, task_name)
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    output_modes = {
        "multiemo": "classification"
    }

    default_params = {
        "multiemo": {"max_seq_length": 128, "batch_size": 16}
    }

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # Prepare seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size * n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
    elif 'multiemo' in task_name:
        args.batch_size = default_params['multiemo']["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size * n_gpu)
        args.max_seq_length = default_params['multiemo']["max_seq_length"]

    if 'multiemo' in task_name:
        _, lang, domain, kind = task_name.split('_')
        processor = MultiemoProcessor(lang, domain, kind)
    else:
        raise ValueError("Task not found: %s" % task_name)

    if 'multiemo' in task_name:
        output_mode = output_modes['multiemo']
    else:
        raise ValueError("Task not found: %s" % task_name)

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=args.do_lower_case)

    #########################
    #       Test model      #
    #########################
    test_examples = processor.get_test_examples(data_dir)
    test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer,
                                                 output_mode)

    test_data, test_labels = get_tensor_data(output_mode, test_features)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    config = BertConfig.from_pretrained(
        model_dir,
        quantize_act=True,
        weight_bits=args.weight_bits,
        input_bits=args.input_bits,
        clip_val=args.clip_val
    )
    model = QuantBertForSequenceClassification.from_pretrained(model_dir, config=config, num_labels=num_labels)
    model.to(device)

    model_quant_dir = os.path.join(model_dir, 'quant')
    qunat_config = BertConfig.from_pretrained(
        model_quant_dir,
        quantize_act=True,
        weight_bits=args.weight_bits,
        input_bits=args.input_bits,
        clip_val=args.clip_val
    )
    quant_model = QuantBertForSequenceClassification.from_pretrained(model_quant_dir, config=qunat_config,
                                                                     num_labels=num_labels)
    quant_model.to(device)

    output_quant_dir = os.path.join(output_dir, 'quant')
    for m, out_dir in zip([model, quant_model], [output_dir, output_quant_dir]):
        logger.info("\n***** Running evaluation on test dataset *****")
        logger.info("  Num examples = %d", len(test_features))
        logger.info("  Batch size = %d", args.batch_size)

        eval_start_time = time.monotonic()
        m.eval()
        result, y_logits = do_eval(m, task_name, test_dataloader,
                                   device, output_mode, test_labels, num_labels)
        eval_end_time = time.monotonic()

        diff = timedelta(seconds=eval_end_time - eval_start_time)
        diff_seconds = diff.total_seconds()
        result['eval_time'] = diff_seconds
        result_to_text_file(result, os.path.join(out_dir, "test_results.txt"))

        y_pred = np.argmax(y_logits, axis=1)
        print('\n\t**** Classification report ****\n')
        print(classification_report(test_labels.numpy(), y_pred, target_names=label_list))

        report = classification_report(test_labels.numpy(), y_pred, target_names=label_list, output_dict=True)
        report['eval_time'] = diff_seconds
        dictionary_to_json(report, os.path.join(out_dir, "test_results.json"))


if __name__ == "__main__":
    main()
