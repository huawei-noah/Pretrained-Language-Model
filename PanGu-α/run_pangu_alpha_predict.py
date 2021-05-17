"""
PanGu predict run
"""

import argparse
from pangu_alpha_config import PANGUALPHAConfig, set_parse
from pangu_alpha_predict import run_predict


if __name__ == "__main__":
    """predict function for PANGUALPHA"""
    parser = argparse.ArgumentParser(description="PANGUALPHA predicting")
    parser.add_argument('--device_id',
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--device_num",
                        type=int,
                        default=128,
                        help="Use device nums, default is 1.")
    parser.add_argument("--distribute",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--seq_length",
                        type=int,
                        default=1024,
                        help="sequence length, default is 1024.")
    parser.add_argument("--vocab_size",
                        type=int,
                        default=40000,
                        help="vocabulary size, default is 40000.")
    parser.add_argument("--embedding_size",
                        type=int,
                        default=16384,
                        help="embedding table size, default is 16384.")
    parser.add_argument("--num_layers",
                        type=int,
                        default=64,
                        help="total layers, default is 64.")
    parser.add_argument("--num_heads",
                        type=int,
                        default=128,
                        help="head size, default is 128.")
    parser.add_argument("--stage_num",
                        type=int,
                        default=4,
                        help="Pipeline stage num, default is 4.")
    parser.add_argument("--micro_size",
                        type=int,
                        default=1,
                        help="Pipeline micro_size, default is 1.")
    parser.add_argument("--load_ckpt_name",
                        type=str,
                        default='PANGUALPHA3.ckpt',
                        help="checkpint file name.")
    parser.add_argument("--load_ckpt_path",
                        type=str,
                        default=None,
                        help="predict file path.")
    parser.add_argument('--data_url',
                        required=False,
                        default=None,
                        help='Location of data.')
    parser.add_argument('--train_url',
                        required=False,
                        default=None,
                        help='Location of training outputs.')
    parser.add_argument("--run_type",
                        type=str,
                        default="predict",
                        choices=["train", "predict"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="2.6B",
                        choices=["200B", "13B", "2.6B", "self_define"],
                        help="The train/eval mode")
    parser.add_argument("--strategy_load_ckpt_path",
                    type=str,
                    default="",
                    help="The training prallel strategy for the model.")
    parser.add_argument("--tokenizer_path",
                    type=str,
                    default="./tokenizer_path",
                    help="The path where stores vocab and vocab model file")

    args_opt = parser.parse_args()
    # The ckpt path shoud like args_opt.load_ckpt_path + f"rank_{rank_id}}/" + args_opt.load_ckpt_name, and the rank_id is the training rank_id.
    set_parse(args_opt)
    run_predict(args_opt)

