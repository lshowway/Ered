import argparse
import util
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'test'])
    parser.add_argument("--task_name", type=str, default="")  #, choices=['quality_control']
    parser.add_argument("--data_dir", type=str, default="", )
    parser.add_argument("--output_dir", type=str, default="", )
    # parser.add_argument("--qid_file", type=str, default="../data/knowledge/pretrain/wikidata_description/all_wikidata5m_QIDs_name_des.wikidata", )
    parser.add_argument("--qid_file", type=str, default="../data/knowledge/pretrain/wikipedia_description/wikidata5m_des.wikipedia")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=256)  # 300

    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--backbone_seq_length", type=int, default=32)
    parser.add_argument("--knowledge_seq_length", type=int, default=64)
    parser.add_argument("--max_ent_num", type=int, default=2)
    parser.add_argument("--max_des_num", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)

    # model training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", default=-1, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=-1)
    parser.add_argument('--save_steps', type=int, default=-1, help="Save checkpoint every X updates steps.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--max_save_checkpoints', type=int, default=20,
                        help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")

    parser.add_argument("--model_type", default="KFormers", type=str)
    parser.add_argument("--backbone_model_type", default="luke", type=str)
    parser.add_argument("--knowledge_model_type", default="distilbert", type=str)
    parser.add_argument("--backbone_model_name_or_path", default="bert-base-uncased", type=str, )
    parser.add_argument("--knowledge_model_name_or_path", default="distilbert-base-uncased", type=str, )
    parser.add_argument("--post_trained_checkpoint", default=None, type=str)
    parser.add_argument("--post_trained_checkpoint_embedding", default="G:\D\MSRA\knowledge_aware\checkpoints\checkpoint-280000", type=str)
    t = dict(zip(range(24), range(24)))
    # parser.add_argument("--backbone_knowledge_dict", default={0: 0, 5: 1, 8: 2, 9: 3, 10: 4, 22: 5}, type=dict)
    #parser.add_argument("--backbone_knowledge_dict", default='{18: 0, 19: 1, 20: 2, 21: 3, 22: 4, 23: 5}', type=str)
    parser.add_argument("--backbone_knowledge_dict", default='{23: -1, 22: -1, 21:-1}', type=str)
    parser.add_argument("--add_knowledge", type=util.str2bool, default=True)

    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--fp16", type=util.str2bool, default=False)
    parser.add_argument("--update_K_module", type=util.str2bool, default=False)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # parser.add_argument("--use_entity", type=utils.str2bool, default=False)
    # parser.add_argument('--negative_sample', type=int, default=45000, help='how many negative samples to select')

    args = parser.parse_args()
    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
