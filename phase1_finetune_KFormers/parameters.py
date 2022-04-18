import argparse
import utils
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
    parser.add_argument("--task_name", type=str, default="")  #, choices=['quality_control']
    parser.add_argument("--data_dir", type=str, default="", )
    parser.add_argument("--output_dir", type=str, default="", )
    parser.add_argument("--qid_file", type=str, default="../data/knowledge/all_entity_typing_QIDs_Ename_des.wikipedia", )
    parser.add_argument("--entity_vocab_file", type=str, default="../data/knowledge/entity_qid_vocab.tsv", )

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=30)  # 300
    parser.add_argument("--test_batch_size", type=int, default=300)

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--backbone_seq_length", type=int, default=32)
    parser.add_argument("--knowledge_seq_length", type=int, default=64)
    parser.add_argument("--max_num_entity", type=int, default=2)

    # model training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=-1)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--model_type", default="KFormers", type=str)
    parser.add_argument("--backbone_model_type", default="bert", type=str)
    parser.add_argument("--knowledge_model_type", default="distilbert", type=str)
    parser.add_argument("--backbone_model_name_or_path", default="bert-base-uncased", type=str, )
    parser.add_argument("--knowledge_model_name_or_path", default="distilbert-base-uncased", type=str, )
    parser.add_argument("--post_trained_checkpoint", default=None, type=str)
    # parser.add_argument("--post_trained_checkpoint_embedding", default="../phase2_pretrain_KFormers/output_update_6_40_60-90/checkpoint-260000/", type=str)
    parser.add_argument("--post_trained_checkpoint_embedding", default="G:\D\MSRA\knowledge_aware\checkpoints\checkpoint-280000", type=str)

    parser.add_argument("--backbone_knowledge_dict", default={0: 0, 1: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5}, type=dict)
    # parser.add_argument("--backbone_knowledge_dict", default={0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5}, type=dict)
    # parser.add_argument("--backbone_knowledge_dict", default={6: 0, 7: 1, 8: 2, 9: 3, 10: 4, 11: 5}, type=dict)
    # parser.add_argument("--backbone_knowledge_dict",
    #                     default={0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5}, type=dict)
    parser.add_argument("--add_knowledge", type=utils.str2bool, default=True)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fp16", type=utils.str2bool, default=False)
    parser.add_argument("--update_K_module", type=utils.str2bool, default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--use_entity", type=utils.str2bool, default=False)
    parser.add_argument('--negative_sample', type=int, default=45000, help='how many negative samples to select')

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
