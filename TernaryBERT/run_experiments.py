import logging
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

num_train_epochs = 3
learning_rate = 5e-5
weight_decay = 0.01


def main():
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 scripts/download_dataset.py --data_dir data/multiemo2'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased')):
        logger.info("Downloading bert-base-uncased model")
        cmd = 'python3 download_bert_base.py'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased', 'multiemo_en_all_sentence')):
        cmd = 'python3 multiemo_fine_tune_bert.py '
        options = [
            '--pretrained_model', 'data/models/bert-base-uncased',
            '--data_dir', 'data/multiemo2',
            '--task_name', 'multiemo_en_all_sentence',
            '--output_dir', 'data/models/bert-base-uncased/multiemo_en_all_sentence',
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Training bert-base-uncased for multiemo_en_all_sentence")
        run_process(cmd)

    cmd = 'python3 quant_task_multiemo.py '
    options = [
        '--data_dir', 'data/multiemo2',
        '--model_dir ', 'data/models/bert-base-uncased',
        '--task_name', 'multiemo_en_all_sentence',
        '--output_dir', 'data/models/ternarybert',
        '--learning_rate', str(learning_rate),
        '--num_train_epochs', str(num_train_epochs),
        '--weight_decay', str(weight_decay),
        '--weight_bits', str(2),
        '--input_bits', str(8),
        '--pred_distill',
        '--intermediate_distill',
        '--save_fp_model',
        '--save_quantized_model',
        '--do_lower_case'
    ]
    cmd += ' '.join(options)
    logger.info(f"Training ternarybert for multiemo_en_all_sentence")
    run_process(cmd)

    # cmd = f'python3 -m gather_results --task_name multiemo_en_all_sentence'
    # logger.info(f"Gathering results to csv for multiemo_en_all_sentence")
    # run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
