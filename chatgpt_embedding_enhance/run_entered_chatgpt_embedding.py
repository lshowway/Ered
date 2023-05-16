import time
import logging
import os
import torch
import ast
import shutil
import numpy as np
import random
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.models.roberta import RobertaConfig, RobertaForSequenceClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter
from parameters import parse_args
from utils import compute_metrics
from data_utils import output_modes, processors, final_metric

logger = logging.getLogger(__name__)

BACKBONE_MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
}


def do_train(args, model, train_dataset, val_dataset, test_dataset=None):

    args.total_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.total_batch_size)

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad]

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
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999))
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
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="./kformers_runs/", purge_step=global_step)

    train_iterator = trange(args.epochs, desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducebility (even between python 2 and 3)
    start_time = time.time()
    best_dev_result = 0.0
    for epoch in train_iterator:
        if epoch > 0:
            train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        epoch_iterator = tqdm(train_dataloader, desc="Training", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            step_start_time = time.time()
            loss = 0.0
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.task_name in ['openentity', 'fewrel']:
                input_ids, input_mask, segment_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                des_embedding = batch[-2]
                label = batch[-1]
            elif args.task_name in ['sst2', 'eem']:
                input_ids, input_mask, segment_ids = batch[0], batch[1], batch[2]
                des_embedding = batch[-2]
                label = batch[-1]
                start_id = None

            inputs = {"input_ids": input_ids,
                      "attention_mask": input_mask,
                      "token_type_ids": segment_ids,

                      "start_id": start_id,

                      "des_embedding": des_embedding,

                      "labels": label,
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
                del batch
                torch.cuda.empty_cache()
                # logger.info('The training time of one batch is {}'.format(time.time() - step_start_time))
                # print('The training time of one batch is {}'.format(time.time() - step_start_time))
                # ============================================== 以下要写到acc里面，要不然会打印很多遍
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("epoch {}, step {}, global_step {}, train_loss: {:.5f}".format(epoch, step + 1, global_step, loss))
                if args.local_rank in [-1, 0] and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                    eval_results, eval_loss = do_eval(model, args, val_dataset, global_step)
                    test_results, test_loss = do_eval(model, args, test_dataset, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('train_loss', loss, global_step)
                    tb_writer.add_scalar('dev_loss', eval_loss, global_step)
                    tb_writer.add_scalar('test_loss', test_loss, global_step)
                    t = eval_results[final_metric[args.task_name]]
                    if t > best_dev_result:  # f1
                        best_dev_result = eval_results[final_metric[args.task_name]]
                        logger.info('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
                        logger.info('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
                        print('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch, global_step, eval_results))
                        print('epoch: {}, global step: {}, test results: {} **'.format(epoch, global_step, test_results))
                    else:
                        logger.info('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
                        logger.info('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
                        print('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step, eval_results))
                        print('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    logging.info('Saving checkpoint...')
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))  # save checkpoint
                    logger.info("Save model checkpoint, optimizer, scheduler, args, global_step to %s", output_dir)
                    # control the number of checkpoints to save
                    if (global_step / args.save_steps) > args.max_save_checkpoints:
                        try:
                            shutil.rmtree(os.path.join(args.output_dir, 'checkpoint-{}'.format(
                                global_step - args.max_save_checkpoints * args.save_steps)))
                        except OSError as e:
                            logging.error(e)
                # ==============================================

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        # evaluate per epoch
        if args.local_rank in [-1, 0]:
            eval_results, _ = do_eval(model, args, val_dataset, global_step=epoch)
            test_results, _ = do_eval(model, args, test_dataset, global_step=epoch)
            t = eval_results[final_metric[args.task_name]]
            if t > best_dev_result:  # f1
                best_dev_result = eval_results[final_metric[args.task_name]]
                logger.info('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch,
                                                                                             global_step, eval_results))
                # print('rank: {}, epoch: {}, global step: {}, dev results: {}**'.format(args.local_rank, epoch,
                #                                                                        global_step, eval_results))
            else:
                logger.info(
                    'rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step,
                                                                                  eval_results))
                # print('rank: {} epoch: {}, global step: {}, dev results: {}'.format(args.local_rank, epoch, global_step,
                #                                                                     eval_results))
            if args.save_steps > 0:
                logging.info('Saving checkpoint...')
                output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(global_step, test_results[
                    final_metric[args.task_name]]))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training
                torch.save(model_to_save.state_dict(),
                           os.path.join(output_dir, "pytorch_model.bin"))  # save checkpoint
                logger.info("Save model checkpoint to %s", output_dir)

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    if args.local_rank in [-1, 0]:
        tb_writer.close()
    logging.info("train time:{}".format(time.time() - start_time))


def do_eval(model, args, val_dataset, global_step):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.valid_batch_size)

    preds = None
    out_label_ids = None
    total_loss = 0.0
    eval_iterator = tqdm(val_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(eval_iterator):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if args.task_name in ['openentity', 'fewrel']:
                input_ids, input_mask, segment_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                des_embedding = batch[-2]
                label = batch[-1]
            elif args.task_name in ['sst2', 'eem']:
                input_ids, input_mask, segment_ids = batch[0], batch[1], batch[2]
                des_embedding = batch[-2]
                label = batch[-1]
                start_id = None
            with torch.no_grad():
                inputs = {"input_ids": input_ids,
                          "attention_mask": input_mask,
                          "token_type_ids": segment_ids,

                          "start_id": start_id,

                          "des_embedding": des_embedding,

                          "labels": None,
                          }
                step_time_start = time.time()
                logits = model(**inputs)
                # logger.info('The inferring time of one batch is {}-'.format(time.time() - step_time_start))
                # print('The inferring time of one batch is {}-'.format(time.time() - step_time_start))
                del batch
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)

    if args.task_name in ['openentity',  'figer', 'sst2', 'eem']:
        pass
    elif args.task_name in ['tacred', 'fewrel']:
        preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task_name, preds, out_label_ids)
    del val_dataloader
    del val_sampler
    return result, total_loss


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    args = parse_args()
    if args.task_name in ['openentity']:
        from chat_enhance_modeling import EntityTyping as KFormersDownstreamModel
    elif args.task_name in ['fewrel']:
        from chat_enhance_modeling import RelationClassification as KFormersDownstreamModel
    elif args.task_name in ['sst2']:
        from chat_enhance_modeling import SequenceClassification as KFormersDownstreamModel
    # elif args.task_name in ['eem']:
    #     from chat_enhance_modeling import KFormersForSequencePairClassification as KFormersDownstreamModel
    else:
        KFormersDownstreamModel = None

    if not args.add_knowledge and args.update_K_module:
        raise ValueError("when knowledge is not used, K-module should be closed")
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.backbone_knowledge_dict = ast.literal_eval(args.backbone_knowledge_dict) if isinstance(
        args.backbone_knowledge_dict, str) else args.backbone_knowledge_dict

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG if args.local_rank in [-1, 0] else logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    backbone_config_class, backbone_model_class, backbone_tokenizer_class = BACKBONE_MODEL_CLASSES[
        args.backbone_model_type]

    backbone_tokenizer = backbone_tokenizer_class.from_pretrained(args.backbone_model_name_or_path)

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    args.output_mode = output_modes[args.task_name]

    config = backbone_config_class.from_pretrained(args.backbone_model_name_or_path)
    config.backbone_knowledge_dict = args.backbone_knowledge_dict
    config.num_labels = num_labels
    backbone_model = KFormersDownstreamModel.from_pretrained(args.backbone_model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logging.info('loading backbone model: {}, knowledge module model: {}'.format(args.backbone_model_type,
                                                                                 args.knowledge_model_type))
    backbone_model.to(device)

    # ## Training
    if args.mode == 'train':
        from data_utils import load_and_cache_examples
        if args.task_name in ['sst2', 'eem']:
            # there is no test data in sst2(glue) and eem
            test_dataset = load_and_cache_examples(args, processor, backbone_tokenizer,
                                                   dataset_type='dev', evaluate=True)
        else:
            test_dataset = load_and_cache_examples(args, processor, backbone_tokenizer,
                                                   dataset_type='test', evaluate=True)
        if args.task_name in ['fewrel']:
            val_dataset = load_and_cache_examples(args, processor, backbone_tokenizer,
                                                  dataset_type='test', evaluate=True)
        else:
            val_dataset = load_and_cache_examples(args, processor, backbone_tokenizer,
                                                  dataset_type='dev', evaluate=True)
        train_dataset = load_and_cache_examples(args, processor, backbone_tokenizer,
                                                dataset_type='train', evaluate=False)

        # print('backbone_seq_length=', args.backbone_seq_length, 'max_des_num=', args.max_des_num,
        #       ', train_batch_size=', args.train_batch_size,
        #       ', learning_rate=', args.learning_rate, ', alpha, beta=', args.alpha,
        #       args.beta, ', seed=', args.seed)

        logging.info(
            'backbone_seq_length={}, max_des_num={}, '
            'train_batch_size={}, learning_rate={}, alpha, beta={}, {}, seed={}'.format(
                args.backbone_seq_length,
                args.max_des_num,
                args.train_batch_size, args.learning_rate, args.alpha, args.beta,
                args.seed))

        do_train(args, backbone_model, train_dataset, val_dataset, test_dataset)


if __name__ == "__main__":
    main()