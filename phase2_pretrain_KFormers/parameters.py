import sys, os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from argparse import ArgumentParser
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    # model config
    parser.add_argument("--model_type", default='roberta-base', type=str, required=False,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    # data config
    parser.add_argument("--data_dir", default="G:\D\MSRA\knowledge_aware\data\knowledge\pretrain\wikidata_description", type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument('--output_dir', type=Path, default="output/")

    # run config
    parser.add_argument("--max_seq_length", type=int, default=32, help="max lenght of token sequence")
    # parser.add_argument("--num_neg_sample", type=int, default=10, help="max lenght of token sequence")
    parser.add_argument("--entity_emb_size", type=int, default=32, help="the dimension of entity embeddings")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--evaluate_during_training", type=bool, default=False,
    #                     help="Rul evaluation during training at each logging step.")
    # parser.add_argument("--do_lower_case", action='store_true',
    #                     help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float,
    #                     help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)")
    parser.add_argument('--save_steps', type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument('--eval_steps', type=int, default=None, help="eval every X updates steps.")
    parser.add_argument('--max_save_checkpoints', type=int, default=10,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
    # parser.add_argument("--eval_all_checkpoints", action='store_true',
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    # parser.add_argument('--overwrite_output_dir', action='store_true',
    #                     help="Overwrite the content of the output directory")
    # parser.add_argument('--overwrite_cache', action='store_true',
    #                     help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")



    # parser.add_argument("--comment", default='', type=str,
    #                     help="The comment")
    #
    # parser.add_argument("--restore", type=bool, default=True,
    #                     help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")

    args = parser.parse_args()

    return args
