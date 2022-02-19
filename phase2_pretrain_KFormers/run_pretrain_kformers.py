
from parameters import get_args

import torch
import random
import torch.nn as nn
import json
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
import sys, os
import shutil
from tqdm import tqdm, trange

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
import logging
import os
from collections import OrderedDict
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
import time

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForPreTraining,
)
from data_utils import load_and_cache_examples
from KFormers_pretrain_modeling import KModulePretrainingModel
logger = logging.getLogger(__name__)


def load_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = KModulePretrainingModel(config)  # this is KModule
    Kmodule_state_dict = model.state_dict()  # KFormers的全部参数

    # initialize with roberta
    pretrained_state_dict = AutoModelForPreTraining.from_pretrained(args.model_name_or_path).state_dict()
    news_k_state_dict = OrderedDict()
    for key, value in pretrained_state_dict.items():
        news_k_state_dict[key] = value
    Kmodule_state_dict.update(news_k_state_dict)
    model.load_state_dict(state_dict=Kmodule_state_dict)

    return model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def do_eval(model, args, val_dataset, global_step):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.valid_batch_size)

    preds = None
    out_label_ids = None
    eval_iterator = tqdm(val_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])
    total_loss, total_step = 0.0, 0
    for step, batch in enumerate(eval_iterator):
        total_step += 1
        if step > 3:
            break

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
            entity_ids, entity_labels = batch[-2], batch[-1]

            inputs = {"description_ids": input_ids,
                      "description_attention_mask": attention_mask,
                      "description_segment_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                      "entity_ids": entity_ids,
                      "entity_labels": entity_labels,
                      }

            logits, eval_loss = model(**inputs)
            total_loss += eval_loss.item()
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = entity_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, entity_labels.detach().cpu().numpy(), axis=0)

    if args.task_name in ['open_entity',  'figer']:
        pass
    elif args.task_name in ['tacred', 'fewrel']:
        preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task_name, preds, out_label_ids)
    result['eval_loss'] = round(total_loss / total_step, 8)
    return result


def do_train(args, model, train_dataset, val_dataset, test_dataset=None):
    args.total_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.total_batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.max_steps > 0:
        t_total = args.max_steps
        args.epochs = args.max_steps // (len(train_dataloader)) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.warmup_steps > -1:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = t_total * 0.1
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total, last_epoch=-1)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:  # DP方式
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:  # DDP方式
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader) * args.total_batch_size)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Number of GPU = %d", args.n_gpu)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * args.n_gpu)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    train_iterator = trange(args.epochs, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducebility (even between python 2 and 3)
    start_time = time.time()
    best_dev_result = 0.0
    for epoch in train_iterator:
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        epoch_iterator = tqdm(train_dataloader, desc="Training", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            loss = 0.0
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
            entity_ids, entity_labels = batch[-2], batch[-1]

            inputs = {"description_ids": input_ids,
                      "description_attention_mask": attention_mask,
                      "description_segment_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                      "entity_ids": entity_ids,
                      "entity_labels": entity_labels,
                      }

            _, batch_loss = model(**inputs)
            if args.n_gpu > 1:
                batch_loss = batch_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                batch_loss = batch_loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            loss += batch_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logger.info("epoch {}, step {}, global_step {}, train_loss: {:.5f}".format(epoch, step + 1, global_step, loss))
            if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                eval_results = do_eval(model, args, val_dataset, global_step)
                t = eval_results["eval_loss"]
                if t > best_dev_result:
                    best_dev_result = eval_results["eval_loss"]
                    logger.info('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                    print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                else:
                    logger.info('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
                    print('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        # evaluate per epoch
        if args.local_rank in [-1, 0]:
            eval_results = do_eval(model, args, val_dataset, global_step)
            t = eval_results["eval_loss"]
            if t > best_dev_result:
                best_dev_result = eval_results["eval_loss"]
                logger.info('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
            else:
                logger.info('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
                print('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    logging.info("train time:{}".format(time.time() - start_time))


def main():
    args = get_args()
    if args.eval_steps is None:
        args.eval_steps = args.save_steps * 10

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load model
    model = load_model(args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model.to(args.device)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'train', data_dir=args.data_dir, evaluate=False)
    val_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'dev', data_dir=args.data_dir, evaluate=True)

    do_train(args, model, train_dataset, val_dataset)


if __name__ == '__main__':
    main()













