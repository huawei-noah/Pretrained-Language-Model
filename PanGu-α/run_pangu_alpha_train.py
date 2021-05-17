"""
PanGu predict run
"""

import argparse
from pangu_alpha_config import PANGUALPHAConfig, set_parse
from pangu_alpha_train import run_train


if __name__ == "__main__":
    """train function for PanGu-Alpha"""
    parser = argparse.ArgumentParser(description="PanGu training")
    parser.add_argument('--train_url',
                        required=False,
                        default=None,
                        help='Location of training outputs.')
    parser.add_argument('--data_url',
                        required=False,
                        default="/cache_pangu_alpha/V1-sample60-baike-math-bpe-1024",
                        help='Location of data.')
    parser.add_argument("--distribute",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        choices=["adam", "lamb"],
                        help="select which optimizer to be used, default adam")
    parser.add_argument("--epoch_size",
                        type=int,
                        default=1,
                        help="Epoch size, default is 1.")
    parser.add_argument("--warmup_step",
                        type=int,
                        default=2000,
                        help="Warmup step, default is 2000.")
    parser.add_argument("--decay_steps",
                        type=int,
                        default=80000,
                        help="Learning rate decay step, default is 80000.")
    parser.add_argument("--start_lr",
                        type=float,
                        default="6e-5",
                        help="Start learning rate, default is 6e-5.")
    parser.add_argument("--end_lr",
                        type=float,
                        default="6e-6",
                        help="End learning rate, default is 6e-6.")
    parser.add_argument("--sink_size",
                        type=int,
                        default=2,
                        help="Sink size for every iteration, default is 2")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-1,
                        help="weight decay of optimizer")
    parser.add_argument('--ckpt_save_sir',
                        required=False,
                        default="/cache/ckpt/",
                        help='Dir to save ckpt.')
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
    parser.add_argument("--optimizer_shard",
                        type=int,
                        default=0,
                        choices=[0, 1],
                        help="enable optimizer shard.")
    parser.add_argument("--stage_num",
                        type=int,
                        default=16,
                        help="Pipeline stage num, default is 16.")
    parser.add_argument("--micro_size",
                        type=int,
                        default=32,
                        help="Pipeline micro_size, default is 32.")
    parser.add_argument("--tensor_model_parallel_num",
                        type=int,
                        default=16,
                        help="The model parallel dim of slicing tensor.")
    parser.add_argument("--per_batch_size",
                        type=int,
                        default=1,
                        help="The batch size of each card.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=1000,
                        help="Checkpoint save steps, default is 2000.")
    parser.add_argument("--run_type",
                        type=str,
                        default="train",
                        choices=["train", "predict"],
                        help="The run type")
    parser.add_argument("--mode",
                        type=str,
                        default="2.6B",
                        choices=["200B", "13B", "2.6B", "self_define"],
                        help="The train/eval mode")

    args_opt = parser.parse_args()
    # set the input configs by train_mode
    set_parse(args_opt)
    run_train(args_opt)

