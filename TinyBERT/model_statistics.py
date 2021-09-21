from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import sys

import torch
from thop import profile

from data_processing import processors
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler('debug_layer_loss.log')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logger = logging.getLogger()


def print_results(macs, params, title=''):
    if len(title) != 0:
        print("- " + title)
    print(f"\tmacs [G]: {macs / math.pow(10, 9):.2f}, params [M]: {params / math.pow(10, 6):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        help="The anlised model dir.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()
    logger.info('The args: {}'.format(args))

    # Prepare devices
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)

    model = TinyBertForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
    model.to(device)

    model_input = tuple([torch.randint(high=len(tokenizer.vocab),
                                       size=(1, args.max_seq_length), dtype=torch.int64, device=device),
                         torch.randint(high=1, size=(1, args.max_seq_length), dtype=torch.int64, device=device),
                         torch.randint(high=1, size=(1, args.max_seq_length), dtype=torch.int64, device=device)])

    macs, params = profile(model, inputs=model_input)

    print("Results")
    print_results(macs, params)


if __name__ == "__main__":
    main()
