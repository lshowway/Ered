import argparse
import util
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../data/knowledge/tacred", )
    parser.add_argument("--checkpoint_file", default="../checkpoints/luke_large_500k", type=str)
    parser.add_argument("--output_dir", type=str, default="./output_tacred", )
    parser.add_argument("--baseline_model_name", default='roberta-large', type=str, )
    #
    parser.add_argument("--do_train", action='store_true',  help="Whether to run training.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=30)  # 300
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    #
    parser.add_argument("--max_mention_length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--save_steps", type=int, default=-1)
    #
    parser.add_argument("--lr_schedule", default='warmup_linear', type=str,  help="learning rate schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_proportion", default=0.06, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=0.0, type=float, help="Max gradient norm.")
    #
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fp16", type=util.str2bool, default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--fp16_min_loss_scale", type=int, default=1)
    parser.add_argument("--fp16_max_loss_scale", type=int, default=4)
    #
    parser.add_argument("--adam_eps", default=1e-06, type=float,  help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_b1", default=0.9, type=float,  help="b1 for Adam optimizer.")
    parser.add_argument("--adam_b2", default=0.8, type=float,  help="b2 for Adam optimizer.")
    parser.add_argument("--experiment_logger", default="", type=str,  help="experiment logger")
    parser.add_argument("--adam_correct_bias", type=util.str2bool, default=False)
    #
    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
