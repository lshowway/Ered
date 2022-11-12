from __future__ import absolute_import, division, print_function

import logging
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers import (BertConfig, BertForSequenceClassification, BertTokenizer)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from parameters import get_args
from data_utils import (output_modes, processors, load_and_cache_examples, final_metric)
from utils import compute_metrics
from modeling import RoBERTaForEntityTyping, RoBERTaForRelationClassification


logger = logging.getLogger(__name__)



TASK_MODEL = {
    'sst2': {
        'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
        'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    },
    'openentity': {
        'roberta': (RobertaConfig, RoBERTaForEntityTyping, RobertaTokenizer),
    },
    'figer': {
        'roberta': (RobertaConfig, RoBERTaForEntityTyping, RobertaTokenizer),
    },
    'fewrel': {
        'roberta': (RobertaConfig, RoBERTaForRelationClassification, RobertaTokenizer),
    },
    'tacred': {
        'roberta': (RobertaConfig, RoBERTaForRelationClassification, RobertaTokenizer),
    },
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, eval_dataset):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
        eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    preds = None
    out_label_ids = None
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.task_name in ['sst2']:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': None # batch[3]
                          }
                labels = batch[-1]
            elif args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'start_ids': batch[3],
                    'labels': None #batch[-1]
                }
                labels = batch[-1]
            else:
                inputs = None
            step_time_start = time.time()
            logits = model(**inputs)
            logger.info('The [inferring] time of one batch is {}'.format(time.time() - step_time_start))
            print('The [inferring] time of one batch is {}'.format(time.time() - step_time_start))
            if isinstance(logits, SequenceClassifierOutput):
                logits = logits.logits
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    if args.task_name in ['openentity',  'figer', 'sst2']:
        pass
    elif args.task_name in ['tacred', 'fewrel', 'fewrel', 'tacred']:
        preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task_name, preds, out_label_ids)
    del eval_dataloader
    del eval_sampler
    return result


def train(args, model, train_dataset, eval_dataset, test_dataset):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size // args.gradient_accumulation_steps)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_steps > -1:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = t_total * 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total, last_epoch=-1)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_result = 0.0
    model.zero_grad()
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="../baseline_runs/")
    train_iterator = trange(int(args.num_train_epochs), desc="Training Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Training Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            step_time_start = time.time()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.task_name in ["sst2"]:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3], }
            elif args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                    'start_ids': batch[3],
                    'labels': batch[-1]
                }
            else:
                inputs = None
            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info('The [training] time of one batch is {}'.format(time.time() - step_time_start))
                print('The [training] time of one batch is {}'.format(time.time() - step_time_start))
                if args.local_rank in [-1,0] and args.evaluate_steps > 0 and global_step % args.evaluate_steps == 0:
                    eval_results = evaluate(args, model, eval_dataset)
                    test_results = evaluate(args, model, test_dataset)

                    t = eval_results[final_metric[args.task_name]]
                    if t > best_dev_result:
                        best_dev_result = eval_results[final_metric[args.task_name]]
                        # logger.info('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
                        # logger.info('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
                        print('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
                        print('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
                    else:
                        # logger.info('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
                        # logger.info('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
                        print('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
                        print('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        eval_results = evaluate(args, model, eval_dataset)
        test_results = evaluate(args, model, test_dataset)

        t = eval_results[final_metric[args.task_name]]
        if t > best_dev_result:
            best_dev_result = eval_results[final_metric[args.task_name]]
            # logger.info('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
            # logger.info('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
            print('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
            print('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
        else:
            # logger.info('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
            # logger.info('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
            print('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
            print('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def main():
    args = get_args()

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    args.output_mode = output_modes[args.task_name]

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    MODEL_CLASSES = TASK_MODEL[args.task_name]
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path)

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    config.num_labels = num_labels

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()
    logger.info("Training/evaluation parameters %s", args)
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,
                                                tokenizer, 'train', evaluate=False)
        eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'dev', evaluate=True)
        if args.task_name in ['sst2']:
            test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'dev', evaluate=True)
        else:
            test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'test', evaluate=True)
        global_step, tr_loss = train(args, model, train_dataset, eval_dataset, test_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
