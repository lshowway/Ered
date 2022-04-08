import sys, os
from tqdm import tqdm, trange
import time
import random
import shutil

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import logging
from collections import OrderedDict



from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import (
    RobertaTokenizer, BertTokenizer,
    RobertaTokenizerFast, BertTokenizerFast,
    AutoConfig,
    AutoTokenizer,
    AutoModelForPreTraining,
)
from parameters import get_args
from data_utils import load_and_cache_examples, EntityPredictionProcessor
from utils import compute_metrics
from KFormers_pretrain_modeling import KModulePretrainingModel


logger = logging.getLogger(__name__)


def load_model(args):
    logging.info("Loading Model..")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # add K config
    config.entity_vocab_size = args.entity_vocab_size
    config.entity_emb_size = args.entity_emb_size


    model = KModulePretrainingModel(config)  # this is KModule
    Kmodule_state_dict = model.state_dict()  # KFormers的全部参数

    # initialize with roberta
    if args.post_trained_checkpoint is not None:
        pretrained_state_dict = AutoModelForPreTraining.from_pretrained(args.post_trained_checkpoint).state_dict()
    else:
        pretrained_state_dict = AutoModelForPreTraining.from_pretrained(args.model_name_or_path).state_dict()
    news_k_state_dict = OrderedDict()
    for key, value in pretrained_state_dict.items():
        if key in Kmodule_state_dict:
            news_k_state_dict[key] = value

    Kmodule_state_dict.update(news_k_state_dict)
    # == 在这里设置backbone的参数更新情况 == 只更新前六层
    for k, v in Kmodule_state_dict.items():
        v.requires_grad = False
        if 'roberta.encoder.layer.' in k:
            t1 = k.split('roberta.encoder.layer.')[1]
            t2 = t1.split('.')[0]
            t = 'roberta.encoder.layer.' + t2
            if t in ['roberta.encoder.layer.0', 'roberta.encoder.layer.1',
                     'roberta.encoder.layer.2', 'roberta.encoder.layer.3',
                     'roberta.encoder.layer.4', 'roberta.encoder.layer.5']:  # []表示参数更新的层
            # if t in ['roberta.encoder.layer.5', 'roberta.encoder.layer.8',
            #          'roberta.encoder.layer.11', 'roberta.encoder.layer.14',
            #          'roberta.encoder.layer.17', 'roberta.encoder.layer.20']:  # []表示参数更新的层
            # if t in []: # 直接不更新
                v.requires_grad = True
        if 'entity_embeddings' in k:
            v.requires_grad = True
    # == 在这里设置backbone的参数更新情况 ==
    model.load_state_dict(state_dict=Kmodule_state_dict)

    return model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def do_eval(model, args, val_dataset, global_step=None, entity_set=None):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.per_gpu_eval_batch_size)

    preds = None
    out_label_ids = None
    eval_iterator = tqdm(val_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])
    total_loss, total_step = 0.0, 0
    for step, batch in enumerate(eval_iterator):
        total_step += 1

        # if step > 3:
        #     break

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
            pos_entities = batch[-1]
            batch_size = pos_entities.size(0)
            neg_entities = torch.tensor(random.sample(entity_set, args.num_neg_sample * batch_size))
            neg_entities = neg_entities.reshape(batch_size, args.num_neg_sample).to(args.device)
            candidate_entities = torch.cat([pos_entities.reshape(batch_size, -1), neg_entities], dim=-1)  # 正负样本组成候选
            entity_labels = torch.zeros(candidate_entities.size())
            entity_labels[:, 0] = 1  # 这是候选样本的标签：0/1

            # shuffle: train need, dev not need

            inputs = {"description_ids": input_ids,
                      "description_attention_mask": attention_mask,
                      "description_segment_ids": token_type_ids if args.model_type in ['bert', 'unilm'] else None,
                      "candidate_entities": candidate_entities,
                      "entity_labels": entity_labels,
                      }

            logits, eval_loss, entity_labels = model(**inputs)
            total_loss += eval_loss.item()
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = entity_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, entity_labels.detach().cpu().numpy(), axis=0)

    result = compute_metrics(preds, out_label_ids)
    result['eval_loss'] = round(total_loss / total_step, 8)
    return result


def do_train(args, model, train_dataset, val_dataset=None, test_dataset=None, entity_set=None):
    args.total_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        print('random ~~~~')
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.total_batch_size, drop_last=True, pin_memory=True)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.max_steps > -1:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader)) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # if args.local_rank != -1:
    #     t_total = t_total // torch.distributed.get_world_size()
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
    logger.info("  Num examples = %d", len(train_dataloader) * args.total_batch_size * 4)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Number of GPU = %d", 4)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.per_gpu_train_batch_size * args.gradient_accumulation_steps * 4)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="./runs/" + "KModule", purge_step=global_step)

    train_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducebility (even between python 2 and 3)
    start_time = time.time()

    # 对角矩阵
    mention_entity_labels = torch.eye(args.total_batch_size, device=args.device)
    des_entity_labels = torch.eye(args.total_batch_size, device=args.device)

    # mention_entity_labels = torch.arange(0, args.total_batch_size, device=args.device, dtype=torch.long)
    # des_entity_labels = torch.arange(0, args.total_batch_size, device=args.device, dtype=torch.long)

    for epoch in train_iterator:
        if epoch > 0:
            # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.total_batch_size, drop_last=True, pin_memory=True)
            train_dataloader.sampler.set_epoch(epoch)
        epoch_iterator = tqdm(train_dataloader, desc="Training", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch[0],
                      # "input_mask": batch[1],
                      # "input_segment": token_type_ids if args.model_type in ['bert', 'unilm'] else None,
                      # "input_segment": None,
                      "mention_token_label": batch[1],
                      "mention_span_idx": batch[2],
                      "mention_entity_candidates": batch[3],
                      "des_entity_candidates": batch[4],
                      "mention_entity_labels": mention_entity_labels,  # 这个的存在用于计算二值交叉熵
                      "des_entity_labels": des_entity_labels  # 这个的存在用于计算二值交叉熵
                      }

            batch_loss = model(**inputs)
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
            # loss += batch_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # torch.clear_autocast_cache()
                del inputs
                torch.cuda.empty_cache()
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                t = batch_loss.detach()
                logger.info("epoch {}, step {}, global_step {}, train_loss: {:.5f}".format(epoch, step + 1, global_step, t))
                # print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                # Log metrics
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', t, global_step)
            # if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
            #     eval_results = do_eval(model, args, val_dataset, global_step, entity_set)
            #     t = eval_results["eval_loss"]
            #     if t < best_dev_result:
            #         best_dev_result = eval_results["eval_loss"]
            #         logger.info('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
            #         # print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
            #     else:
            #         logger.info('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
            #         # print('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                logging.info('Saving checkpoint...')
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))  # save checkpoint
                # torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))  # save optimizer
                torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.bin'))  # save scheduler
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))  # save arguments
                torch.save(global_step, os.path.join(args.output_dir, 'global_step.bin'))
                logger.info("Save model checkpoint, optimizer, scheduler, args, global_step to %s", output_dir)
                # control the number of checkpoints to save
                if (global_step / args.save_steps) > args.max_save_checkpoints:
                    try:
                        shutil.rmtree(os.path.join(args.output_dir, 'checkpoint-{}'.format(
                            global_step - args.max_save_checkpoints * args.save_steps)))
                    except OSError as e:
                        logging.error(e)
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        # evaluate per epoch
        # if args.local_rank in [-1, 0]:
        #     eval_results = do_eval(model, args, val_dataset, global_step, entity_set)
        #     t = eval_results["eval_loss"]
        #     if t < best_dev_result:
        #         best_dev_result = eval_results["eval_loss"]
        #         logger.info('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
        #         # print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
        #     else:
        #         logger.info('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
        #         # print('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
            # Save model checkpoint
            # output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
            output_dir = os.path.join(args.output_dir, 'checkpoint-epoch-{}-{}'.format(epoch, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))  # save checkpoint
            # torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.bin'))  # save optimizer, too large
            torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.bin'))  # save scheduler
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))  # save arguments
            torch.save(global_step, os.path.join(args.output_dir, 'global_step.bin'))
            logger.info("Save model checkpoint, optimizer, scheduler, args, global_step to %s", output_dir)
            # control the number of checkpoints to save
            if (global_step / args.save_steps) > args.max_save_checkpoints:
                try:
                    shutil.rmtree(os.path.join(args.output_dir, 'checkpoint-{}'.format(
                        global_step - args.max_save_checkpoints * args.save_steps)))
                except OSError as e:
                    logging.error(e)
        # end all epoches
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    logging.info("train time:{}".format(time.time() - start_time))


def main():
    args = get_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logging.warning("Waiting for debugger attach")
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
    print('00', args.local_rank) # 0 1 2 3
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    # if isinstance(tokenizer, RobertaTokenizer) or isinstance(tokenizer, RobertaTokenizerFast):
    #     special_tokens_dict = {'additional_special_tokens': ['<SOM>', '<EOM>']}
    #     tokenizer.add_special_tokens(special_tokens_dict)
    #     # tokenizer.add_special_tokens({'SOM': '<SOM>', 'EOM': '<EOM>'})
    #     # dict(additional_special_tokens=[ENTITY_TOKEN])
    #     tokenizer.som = '<SOM>'
    #     tokenizer.eom = '<EOM>'
    # elif isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, BertTokenizerFast):
    #     tokenizer.add_special_tokens({'SOM': '[SOM]', 'EOM': '[EOM]'})

    # load model
    # args.entity_vocab_file = os.path.join(args.data_dir, "entity_qid_v3.tsv")
    processor = EntityPredictionProcessor(args.data_dir, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    entity_set = processor.get_labels()  #
    args.entity_vocab_size = len(entity_set)

    model = load_model(args)
    print('22', args.local_rank) # 0
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model.to(args.device)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    print('33', args.local_rank)  # 0
    # Training
    if args.do_train:
        start, end = 0, 30
        # start, end = 30, 60
        # start, end = 60, 90
        # start, end = 90, 100
        train_dataset = load_and_cache_examples(args, processor, 'train', evaluate=False, start=start, end=end)
        # val_dataset = load_and_cache_examples(args, processor, tokenizer, 'dev', evaluate=True)


        do_train(args, model, train_dataset, val_dataset=None, test_dataset=None, entity_set=entity_set)


if __name__ == '__main__':
    main()












