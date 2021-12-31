# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.tokenization_gpt2 import RobertaTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from utils.glue_utils import processors, GLUE_TASKS_NUM_LABELS, \
    convert_examples_to_features, compute_metrics, output_modes, auc_metrics_tasks

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--hub_path", default=None, type=str,
                        help="The directory where contains modelings, config files and vocab files")
    parser.add_argument("--warmup_checkpoint", default=None, type=str,
                        help="Path of the checkpoint to warm up. ")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained modelings downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--origin_seq_length", default=32, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to do inference.")
    parser.add_argument("--predict_file",
                        default="",
                        type=str,
                        help="the file used to do inference")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--add_weight_decay_on_layer_norm',
                        action='store_true',
                        help="Add weight decay on layer norm")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--eval_per_epoch", default=10000, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--rel_pos_type', default=0, type=int,
                        help="Relative position type (0: no use; 1: attention weight bias; 2: T5).")
    parser.add_argument('--max_rel_pos', default=128, type=int,
                        help="Maximum relative position.")
    parser.add_argument('--rel_pos_bins', default=32, type=int,
                        help="Maximum bins used for relative position.")
    parser.add_argument('--fast_qkv', action='store_true',
                        help="Fast QKV computation.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate of hidden.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate of attention probabilities.")
    parser.add_argument("--task_dropout_prob", default=0.1, type=float,
                        help="Dropout rate of task layer.")
    parser.add_argument("--lr_layerwise_decay", default=1, type=float,
                        help="Layerwisely decay learning rate.")

    parser.add_argument("--clip_data", action='store_true')

    args = parser.parse_args()

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    return args


def master_process(args):
    return args.no_cuda or (args.local_rank == -1) or (torch.distributed.get_rank() == 0)


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def to_predict(model, eval_features, device, global_step, batch_size, get_prediction, task_name, output_dir=None,
               raw_examples=None):
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", batch_size)
    logger.info("  Global steps = {}".format(global_step))
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    if output_modes[task_name] == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long)
    elif output_modes[task_name] == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.float)
    else:
        raise NotImplementedError()
    # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    prediction = []

    preds = None
    out_label_ids = None

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids,
                                  input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        eval_loss += tmp_eval_loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        nb_eval_steps += 1

        if get_prediction:
            prediction.extend(get_prediction(logits.detach().cpu().numpy()))

    output_eval_file = os.path.join(output_dir, "predict_results.txt")
    writer = open(output_eval_file, 'w', encoding='utf-8')
    preds_softmax = softmax(preds)
    i = 0
    for probabilities in preds_softmax:
        if raw_examples is not None:
            writer.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(str(i), raw_examples[i].text_a,
                                                            raw_examples[i].text_b,
                                                            raw_examples[i].label,
                                                            " ".join([str(p) for p in list(probabilities)])))
        i += 1
    writer.close()


def to_eval(model, eval_features, device, global_step, batch_size, get_prediction, task_name, output_dir=None,
            args=None):
    if master_process(args):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", batch_size)
        logger.info("  Global steps = {}".format(global_step))
    all_input_ids = torch.tensor(
        [f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in eval_features], dtype=torch.long)
    if output_modes[task_name] == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.long)
    elif output_modes[task_name] == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in eval_features], dtype=torch.float)
    else:
        raise NotImplementedError()
    # all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    prediction = []

    preds = None
    out_label_ids = None

    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids,
                                  input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        eval_loss += tmp_eval_loss.mean().item()

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)

        nb_eval_steps += 1

        if get_prediction:
            prediction.extend(get_prediction(logits.detach().cpu().numpy()))

    pred = []
    if output_modes[task_name] == "classification":
        preds = softmax(preds)
        for probabilities in preds:
            pred.append(1 - probabilities[0])
    elif output_modes[task_name] == "regression":
        for probabilities in preds:
            pred.append(probabilities[0])
    else:
        raise NotImplementedError()

    if master_process(args):
        output_eval_file = os.path.join(output_dir, "eval_results_{0}.txt".format(global_step))
        writer = open(output_eval_file, 'w', encoding='utf-8')
        for probabilities in pred:
            writer.write('{0}\n'.format(probabilities))
        writer.close()

    if output_modes[task_name] == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_modes[task_name] == "regression":
        preds = np.squeeze(preds)
    if task_name.lower() in auc_metrics_tasks:
        result = compute_metrics(task_name, pred, out_label_ids)
    else:
        result = compute_metrics(task_name, preds, out_label_ids)

    result["global_step"] = global_step
    eval_loss = eval_loss / nb_eval_steps
    result = {
        'eval_loss': eval_loss,
        'eval_accuracy': result,
        'global_step': global_step,
    }

    if master_process(args):
        from sklearn import metrics
        y = np.array(out_label_ids)
        pred = np.array(pred)
        auc = 0
        try:
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
        except:
            pass
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(global_step))
            writer.write("{0}\n".format(global_step))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            logger.info("auc={}".format(auc))
            writer.write("%s = %s\n" % ('AUC', str(auc)))

    return result, prediction


def load_model(args, warmup_checkpoint, num_labels, is_training):
    model, _ = BertForSequenceClassification.from_pretrained(
        args.bert_model, hub_path=args.hub_path, warmup_checkpoint=warmup_checkpoint,
        remove_task_specifical_layers=False, num_labels=num_labels,
        no_segment="roberta" in args.bert_model,
        rel_pos_type=args.rel_pos_type, max_rel_pos=args.max_rel_pos, rel_pos_bins=args.rel_pos_bins,
        fast_qkv=args.fast_qkv,
        hidden_dropout_prob=args.hidden_dropout_prob, attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        task_dropout_prob=args.task_dropout_prob,
        cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))

    return model


def main():
    args = get_args()

    args.output_dir = args.output_dir.replace('[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    is_roberta = "roberta" in args.bert_model

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #   raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % task_name)

    processor = processors[task_name]()
    num_labels = GLUE_TASKS_NUM_LABELS[task_name]
    label_list = processor.get_labels()
    output_mode = output_modes[task_name]

    logger.info("***** Running Config *****")
    logger.info("  Task = {}".format(task_name))
    logger.info("  Number of labels = %d", num_labels)
    if output_mode == "classification":
        logger.info("  Label list = [{}]".format(str(label_list)))

    if is_roberta:
        tokenizer = RobertaTokenizer.from_pretrained(args.vocab_file)
    else:
        tokenizer = BertTokenizer.from_pretrained(
            args.vocab_file, do_lower_case=args.do_lower_case, hub_path=args.hub_path)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = load_model(args, args.warmup_checkpoint,
                       num_labels=num_labels, is_training=True)
    num_hidden_layers = model.config.num_hidden_layers
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    weight_decay = args.weight_decay
    logger.info("Set weight_decay as %.4f" % weight_decay)
    if args.lr_layerwise_decay != 1:
        logger.info("Layerwise LR decay={}".format(args.lr_layerwise_decay))
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append(
            {'params': [p for n, p in param_optimizer if (n.startswith("bert.encoder.layer.0.") or not n.startswith(
                "bert.encoder.layer."))], 'weight_decay': weight_decay})
        for i in range(1, num_hidden_layers):
            optimizer_grouped_parameters.append({'params': [p for n, p in param_optimizer if (
                    "bert.encoder.layer.{}.".format(i) in n)], 'weight_decay': weight_decay})
    elif args.add_weight_decay_on_layer_norm:
        logger.info("Add weight decay on all parameters!")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer],
             'weight_decay': weight_decay},
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(
                optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    dev_eval_features = None
    if master_process(args):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
        summary_writer = SummaryWriter(os.path.join(args.output_dir, "log"))

    if args.do_train:
        dev_eval_examples = processor.get_dev_examples(args.data_dir)
        dev_eval_features = convert_examples_to_features(args,
            dev_eval_examples, args.origin_seq_length, args.max_seq_length, tokenizer,
            cls_token='<s>' if is_roberta else '[CLS]',
            sep_token='</s>' if is_roberta else '[SEP]',
            output_mode=output_mode,
        )

        train_features = convert_examples_to_features(args,
            train_examples, args.origin_seq_length, args.max_seq_length, tokenizer,
            cls_token='<s>' if is_roberta else '[CLS]',
            sep_token='</s>' if is_roberta else '[SEP]',
            output_mode=output_mode,
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.float)
        else:
            raise NotImplementedError()

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_step = max(1, len(train_features) //
                        args.train_batch_size // args.eval_per_epoch // args.gradient_accumulation_steps)
        if args.local_rank != -1:
            eval_step = eval_step // torch.distributed.get_world_size()
        logger.info("  Eval steps = %d", eval_step)

        # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        # perf_writer = open(output_eval_file, "w")

        model.train()
        best_perf = None
        total_steps = 0
        for _ in range(int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if master_process(args):
                    summary_writer.add_scalar('loss/ClassifierLoss', loss, global_step)
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                total_steps += 1
                if total_steps % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * \
                                       warmup_linear(
                                           global_step / num_train_optimization_steps, args.warmup_proportion)
                        if args.lr_layerwise_decay == 1:
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        else:
                            _lr = lr_this_step
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = _lr
                                _lr *= args.lr_layerwise_decay
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % eval_step == 0 or global_step == num_train_optimization_steps:
                        result, _ = to_eval(
                            model=model, eval_features=dev_eval_features, device=device, global_step=global_step,
                            batch_size=args.eval_batch_size, get_prediction=None, task_name=task_name,
                            output_dir=args.output_dir, args=args
                        )

                        # perf_writer.write(json.dumps(result, indent=None))
                        # perf_writer.write('\n')
                        # perf_writer.flush()
                        if master_process(args):
                            logger.info("***** Eval results *****")
                            logger.info(json.dumps(result, indent=2))
                        model.train()

                        if master_process(args):
                            # if best_perf is None or isinstance(result["eval_accuracy"], dict) or best_perf < result["eval_accuracy"]:
                            model_to_save = model.module if hasattr(
                                model, 'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(
                                args.output_dir, task_name + str(global_step) + WEIGHTS_NAME)
                            torch.save(model_to_save.state_dict(),
                                       output_model_file)

                            output_config_file = os.path.join(
                                args.output_dir, CONFIG_NAME)
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())

                        if (best_perf is not None) and ("roc_auc" in best_perf):
                            best_perf = result["eval_accuracy"] if result["eval_accuracy"]["roc_auc"] > best_perf[
                                "roc_auc"] else best_perf
                        else:
                            best_perf = result["eval_accuracy"]

        if master_process(args):
            logger.info("Best performance = %s" % json.dumps(best_perf, indent=2))
            if "global_step" in best_perf:
                onlyfiles = [f for f in os.listdir(args.output_dir) if
                             os.path.isfile(os.path.join(args.output_dir, f)) and 'pytorch_model.bin' in f]
                print(onlyfiles)
                for ckpt in onlyfiles:
                    ckpt_step = ckpt[len(args.task_name):-len("pytorch_model.bin")]
                    print("ckpt={}, ckpt_step={}, best_step={}".format(ckpt, ckpt_step, best_perf["global_step"]))
                    if int(ckpt_step) != int(best_perf["global_step"]):
                        os.remove(os.path.join(args.output_dir, ckpt))
                with open(os.path.join(args.output_dir, "best_perf.json"), 'w') as fp_bestperf:
                    json.dump(best_perf, fp_bestperf)
        # perf_writer.close()

    if args.do_eval:
        output_model_file = os.path.join(
            args.output_dir, task_name + '.best.' + WEIGHTS_NAME)
        model = load_model(args, output_model_file,
                           num_labels=num_labels, is_training=False)

        model = model.to(device)
        if args.fp16:
            model = model.half()

        if dev_eval_features is None:
            dev_eval_examples = processor.get_dev_examples(args.data_dir)
            dev_eval_features = convert_examples_to_features(args,
                dev_eval_examples, args.origin_seq_length, args.max_seq_length, tokenizer,
                cls_token='<s>' if is_roberta else '[CLS]',
                sep_token='</s>' if is_roberta else '[SEP]',
                output_mode=output_mode,
            )

        dev_result, _ = to_eval(
            model=model, eval_features=dev_eval_features, device=device, task_name=task_name,
            global_step=global_step, batch_size=args.eval_batch_size, get_prediction=None, args=args
        )

        logger.info("***** Dev results *****")
        logger.info(json.dumps(dev_result, indent=2))
        model.eval()
        dev_perf_file = os.path.join(
            args.output_dir, "dev_{}_perf.json".format(task_name))
        with open(dev_perf_file, mode="w") as writer:
            writer.write(json.dumps(dev_result, indent=2))

        test_eval_examples = processor.get_test_examples(args.data_dir)
        test_eval_features = convert_examples_to_features(args,
            test_eval_examples, args.origin_seq_length, args.max_seq_length, tokenizer,
            cls_token='<s>' if is_roberta else '[CLS]',
            sep_token='</s>' if is_roberta else '[SEP]',
            output_mode=output_mode,
        )

        result, predition = to_eval(
            model=model, eval_features=test_eval_features, device=device, task_name=task_name,
            global_step=global_step, batch_size=args.eval_batch_size, get_prediction=processor.get_pred, args=args
        )

        logger.info("***** Test results *****")
        logger.info(json.dumps(result, indent=2))
        model.train()

        prediction_file = os.path.join(
            args.output_dir, "prediction_{}.txt".format(task_name))
        with open(prediction_file, 'w') as writer:
            writer.write("index\tprediction\n")
            for index, label in enumerate(predition):
                writer.write("{}\t{}\n".format(index, label))

    if args.do_predict:
        output_model_file = args.warmup_checkpoint
        model = load_model(args, output_model_file,
                           num_labels=num_labels, is_training=False)

        model = model.to(device)
        if args.fp16:
            model = model.half()
        test_eval_examples = processor.get_predict_examples(args.predict_file)
        test_eval_features = convert_examples_to_features(args,
            test_eval_examples, args.origin_seq_length, args.max_seq_length, tokenizer,
            cls_token='<s>' if is_roberta else '[CLS]',
            sep_token='</s>' if is_roberta else '[SEP]',
            output_mode=output_mode,
        )

        to_predict(
            model=model, eval_features=test_eval_features, device=device, task_name=task_name,
            global_step=global_step, batch_size=args.eval_batch_size, get_prediction=processor.get_pred,
            output_dir=args.output_dir, raw_examples=test_eval_examples
        )

        logger.info("***** Inference Finished *****")


if __name__ == "__main__":
    main()
