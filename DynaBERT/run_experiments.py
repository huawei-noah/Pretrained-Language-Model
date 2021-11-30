import logging
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

batch_size = 16
num_train_epochs = 3
learning_rate = 5e-5
weight_decay = 0.01


def main():
    print(PROJECT_FOLDER)
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
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--weight_decay', str(weight_decay),
            '--train_batch_size', str(batch_size),
            '--do_lower_case'
        ]
        cmd += ' '.join(options)
        logger.info(f"Training bert-base-uncased for multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'dynabertw', 'multiemo_en_all_sentence')):
        cmd = 'python3 run_multiemo.py '
        options = [
            '--model_type', 'bert',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_train',
            '--data_dir', 'data/multiemo2',
            '--model_dir ', 'data/models/bert-base-uncased/multiemo_en_all_sentence',
            '--output_dir', 'data/models/dynabertw/multiemo_en_all_sentence',
            '--max_seq_length', str(128),
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--per_gpu_train_batch_size', str(batch_size),
            '--weight_decay', str(weight_decay),
            '--width_mult_list', '0.25,0.5,0.75,1.0',
            '--width_lambda1', str(1.0),
            '--width_lambda2', str(0.1),
            '--training_phase', 'dynabertw'
        ]
        cmd += ' '.join(options)
        logger.info(f"Training DynaBERT_W for multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'dynabert', 'multiemo_en_all_sentence')):
        cmd = 'python3 run_multiemo.py '
        options = [
            '--model_type', 'bert',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_train',
            '--data_dir', 'data/multiemo2',
            '--model_dir ', 'data/models/dynabertw/multiemo_en_all_sentence',
            '--output_dir', 'data/models/dynabert/multiemo_en_all_sentence',
            '--max_seq_length', str(128),
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--per_gpu_train_batch_size', str(batch_size),
            '--weight_decay', str(weight_decay),
            '--width_mult_list', '0.25,0.5,0.75,1.0',
            '--depth_mult_list', '0.5,0.75,1.0',
            '--width_lambda1', str(1.0),
            '--width_lambda2', str(1.0),
            '--training_phase', 'dynabert',
        ]
        cmd += ' '.join(options)
        logger.info(f"Training DynaBERT for multiemo_en_all_sentence")
        run_process(cmd)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'dynabert-finetuned', 'multiemo_en_all_sentence')):
        cmd = 'python3 run_multiemo.py '
        options = [
            '--model_type', 'bert',
            '--task_name', 'multiemo_en_all_sentence',
            '--do_train',
            '--data_dir', 'data/multiemo2',
            '--model_dir ', 'data/models/dynabertw/multiemo_en_all_sentence',
            '--output_dir', 'data/models/dynabert-finetuned/multiemo_en_all_sentence',
            '--max_seq_length', str(128),
            '--learning_rate', str(learning_rate),
            '--num_train_epochs', str(num_train_epochs),
            '--per_gpu_train_batch_size', str(batch_size),
            '--weight_decay', str(weight_decay),
            '--width_mult_list', '0.25,0.5,0.75,1.0',
            '--depth_mult_list', '0.5,0.75,1.0',
            '--training_phase', 'final_finetuning ',
        ]
        cmd += ' '.join(options)
        logger.info(f"Finetuning DynaBERT for multiemo_en_all_sentence")
        run_process(cmd)


    cmd = 'python3 eval_multiemo.py '
    options = [
        '--model_type', 'bert',
        '--task_name', 'multiemo_en_all_sentence',
        '--data_dir', 'data/multiemo2',
        '--model_dir', 'data/models/dynabert-finetuned/multiemo_en_all_sentence'
        '--output_dir', 'data/models/dynabert-finetuned/multiemo_en_all_sentence'
        '--max_seq_length', str(128),
        '--depth_mult', '0.5'
    ]
    cmd += ' '.join(options)
    logger.info(f"Evaluating DynaBERT for multiemo_en_all_sentence")
    run_process(cmd)


    # cmd = f'python3 -m gather_results --task_name multiemo_en_all_sentence'
    # logger.info(f"Gathering results to csv for multiemo_en_all_sentence")
    # run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
