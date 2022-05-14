import time
import logging
import os
import torch
import shutil
import numpy as np
import random
from collections import OrderedDict
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
# knowledge module
from transformers.models.distilbert import DistilBertConfig, DistilBertForSequenceClassification
from transformers.models.distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
# backbone module
from transformers.models.bert import BertConfig, BertForSequenceClassification
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
# roberta
from transformers.models.roberta import RobertaConfig, RobertaForSequenceClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from transformers.optimization import AdamW, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup

from tensorboardX import SummaryWriter
from parameters import parse_args
from utils import compute_metrics
from data_utils import output_modes, processors, final_metric
from model_utils import ModelArchive
from util import NullLogger
from data_utils import ENTITY_TOKEN, MASK_TOKEN, HEAD_TOKEN, TAIL_TOKEN

logger = logging.getLogger(__name__)

BACKBONE_MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
    'luke': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
}
KNOWLEDGE_MODEL_CLASSES = {
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast),
}



def load_kformers(args, k_config_class, k_model_class, KFormersDownstreamModel):

    # 加载luke的参数
    model_archive = ModelArchive.load("../checkpoints/luke_large_500k")
    args.model_config = model_archive.config

    args.model_config.num_labels = args.num_labels
    args.entity_vocab = model_archive.entity_vocab # 不用这一步了吧

    args.experiment = NullLogger()
    # load checkpoint
    model_weights = torch.load('../checkpoints/luke_large_500k/pytorch_model.bin', map_location=args.device)

    # word embedding
    word_emb = model_weights["embeddings.word_embeddings.weight"]  # 50265*768

    if args.task_name in ['openentity', 'figer']:
        args.model_config.vocab_size += 1
        args.model_config.entity_vocab_size = 2
        args.model_config.k_entity_vocab_size = len(args.entity_vocab)

        # word embedding
        word_emb = model_weights["embeddings.word_embeddings.weight"]  # 50265*768
        marker_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)  # 1*768
        model_weights["embeddings.word_embeddings.weight"] = torch.cat([word_emb, marker_emb])  # 后面拼一个marker_emb
        args.tokenizer.add_special_tokens(dict(additional_special_tokens=[ENTITY_TOKEN]))
        # ent_knowledge embedding
        entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]  # 50W*256
        model_weights['k_ent_embeddings.entity_embeddings.weight'] = entity_emb  # ???还要多拼一个吗？
        model_weights['k_ent_embeddings.entity_embedding_dense.weight'] = model_weights[
            'entity_embeddings.entity_embedding_dense.weight']
        model_weights['k_ent_embeddings.LayerNorm.weight'] = model_weights['entity_embeddings.LayerNorm.weight']
        model_weights['k_ent_embeddings.LayerNorm.bias'] = model_weights['entity_embeddings.LayerNorm.bias']
        # entity embedding
        entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]  # 50W*256
        mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0)  # 1*256
        model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])  # 2*256?
    else:
        args.model_config.vocab_size += 2  # 1 2
        args.model_config.entity_vocab_size = 3
        args.model_config.k_entity_vocab_size = len(args.entity_vocab)

        head_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
        tail_emb = word_emb[args.tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)

        model_weights["embeddings.word_embeddings.weight"] = torch.cat([word_emb, head_emb, tail_emb])  # 50267*1024
        args.tokenizer.add_special_tokens(dict(additional_special_tokens=[HEAD_TOKEN, TAIL_TOKEN]))
        # ent_knowledge embedding
        entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]  # 50W*256
        model_weights['k_ent_embeddings.entity_embeddings.weight'] = entity_emb  # ???还要多拼一个吗？
        model_weights['k_ent_embeddings.entity_embedding_dense.weight'] = model_weights[
            'entity_embeddings.entity_embedding_dense.weight']
        model_weights['k_ent_embeddings.LayerNorm.weight'] = model_weights['entity_embeddings.LayerNorm.weight']
        model_weights['k_ent_embeddings.LayerNorm.bias'] = model_weights['entity_embeddings.LayerNorm.bias']
        # entity embedding
        entity_emb = model_weights["entity_embeddings.entity_embeddings.weight"]  # 50W*256
        mask_emb = entity_emb[args.entity_vocab[MASK_TOKEN]].unsqueeze(0).expand(2, -1) # 2*256
        # model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])  # 2*256?
        model_weights["entity_embeddings.entity_embeddings.weight"] = torch.cat([entity_emb[:1], mask_emb])  # 3*256


    for num in range(args.model_config.num_hidden_layers):
        for attr_name in ("weight", "bias"):
            if f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}" not in model_weights:
                model_weights[f"encoder.layer.{num}.attention.self.w2e_query.{attr_name}"] = model_weights[
                    f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]
            if f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}" not in model_weights:
                model_weights[f"encoder.layer.{num}.attention.self.e2w_query.{attr_name}"] = model_weights[
                    f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]
            if f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}" not in model_weights:
                model_weights[f"encoder.layer.{num}.attention.self.e2e_query.{attr_name}"] = model_weights[
                    f"encoder.layer.{num}.attention.self.query.{attr_name}"
                ]

    #=================================
    config_k = k_config_class.from_pretrained(args.knowledge_model_name_or_path, output_hidden_states=True)  # 这是DistillBert的config
    kformers_model = KFormersDownstreamModel(args=args, config_k=config_k,
                                             backbone_knowledge_dict=args.backbone_knowledge_dict)

    kformer_dict = kformers_model.state_dict()  # KFormers的全部参数


    news_kformers_state_dict = OrderedDict()
    # 加载luke的预训练参数
    for key, value in model_weights.items():
        key = 'kformers.'+key
        if 'embeddings' in key or 'pooler' in key:
            news_kformers_state_dict[key] = value
        elif 'encoder' in key:
            i = key.split('kformers.encoder.layer.')[-1]
            i = i.split('.')[0]
            i = int(i)
            key = key.replace('kformers.encoder.layer.%s' % i, 'kformers.encoder.layer.%s.backbone_layer' % i)
            news_kformers_state_dict[key] = value
        elif 'classifier.dense' in key: # classifier的dense要不要初始化呢？
            news_kformers_state_dict[key] = value
            pass
        else:
            pass
    # 加载distilbert的预训练参数，并把值赋给kformers
    backbone_knowledge_dict_reverse = {v: k for k, v in args.backbone_knowledge_dict.items()}
    knowledge_model_dict = k_model_class.from_pretrained(args.knowledge_model_name_or_path,
                                                         config=config_k).state_dict()
    for key, value in knowledge_model_dict.items():
        key = key.replace(args.knowledge_model_type, 'kformers')
        if 'embeddings' in key:
            key = key.replace('kformers.embeddings', 'kformers.k_des_embeddings')
            news_kformers_state_dict[key] = value
        elif 'layer' in key:
            j = key.split('kformers.transformer.layer.')[-1]
            j = j.split('.')[0]
            i = backbone_knowledge_dict_reverse[int(j)]
            key = key.replace('kformers.transformer.layer.%s' % j, 'kformers.encoder.layer.%s.k_layer' % i)
            news_kformers_state_dict[key] = value
        else:
            pass
    kformer_dict.update(news_kformers_state_dict)
    kformers_model.load_state_dict(state_dict=kformer_dict)

    # == 在这里设置K-module的参数更新不更新 ==
    t = backbone_knowledge_dict_reverse[5]
    if not args.update_K_module:
        for k, v in kformers_model.named_parameters():
            if 'k_' in k and str(t) not in k:
                v.requires_grad = False
    # == 在这里设置K-module的参数更新不更新 ==
    return args, kformers_model


def do_train(args, model, train_dataset, val_dataset, test_dataset=None):
    args.total_batch_size = args.train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.total_batch_size)

    # param_optimizer = list(model.named_parameters())
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    w = 1.2

    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay) and 'k_' in n], 'weight_decay': args.weight_decay, 'lr': args.learning_rate * w},
    #     {'params': [p for n, p in param_optimizer if not any(
    #         nd in n for nd in no_decay) and 'k_' not in n], 'weight_decay': args.weight_decay,
    #      'lr': args.learning_rate},
    #
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay) and 'k_' in n], 'weight_decay': 0.0, 'lr': args.learning_rate * w},
    #     {'params': [p for n, p in param_optimizer if any(
    #         nd in n for nd in no_decay) and 'k_' not in n], 'weight_decay': 0.0, 'lr': args.learning_rate},
    # ]

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
        tb_writer = SummaryWriter(log_dir="./finetune_runs/", purge_step=global_step)

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

            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                input_ids, input_mask, segment_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                ent_ids, ent_mask, ent_seg_ids, ent_pos_ids = batch[4], batch[5], batch[6], batch[7]
                k_ent_ids, k_label = batch[8], batch[9]
                des_input_ids, des_att_mask_one, des_segment_one, des_mask = batch[10], batch[11], batch[12], batch[13]
                label = batch[-1]

            inputs = {"input_ids": input_ids,
                      "attention_mask": input_mask,
                      "token_type_ids": segment_ids,

                      "start_id": start_id,
                      "entity_ids": ent_ids,
                      "entity_position_ids": ent_pos_ids,
                      "entity_segment_ids": ent_seg_ids,
                      "entity_attention_mask": ent_mask,

                      "k_ent_ids": k_ent_ids,
                      "k_label": k_label,

                      "k_des_ids": des_input_ids,
                      "k_des_mask_one": des_att_mask_one,
                      "k_des_seg": des_segment_one,
                      "k_des_mask": des_mask,
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
                    print('epoch: ', epoch, 'global step: ', global_step, 'eval results: ', eval_results, '**')
                    print('epoch: ', epoch, 'global step: ', global_step, 'test results: ', test_results, '**')

                    # logging.info('Saving checkpoint...')
                    # output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(global_step, test_results[final_metric[args.task_name]]))
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    # model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    # torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))  # save checkpoint
                    # logger.info("Save model checkpoint to %s", output_dir)
                else:
                    print('epoch: ', epoch, 'global step: ', global_step, 'eval results: ', eval_results)
                    print('epoch: ', epoch, 'global step: ', global_step, 'test results: ', test_results)
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
                print('epoch: ', epoch, 'global step: ', global_step, 'eval results: ', eval_results, '**')
                print('epoch: ', epoch, 'global step: ', global_step, 'test results: ', test_results, '**')
            else:
                print('epoch: ', epoch, 'global step: ', global_step, 'eval results: ', eval_results)
                print('epoch: ', epoch, 'global step: ', global_step, 'test results: ', test_results)

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
            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                input_ids, input_mask, segment_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                ent_ids, ent_mask, ent_seg_ids, ent_pos_ids = batch[4], batch[5], batch[6], batch[7]
                k_ent_ids, k_label = batch[8], batch[9]
                des_input_ids, des_att_mask_one, des_segment_one, des_mask = batch[10], batch[11], batch[12], batch[13]
                labels = batch[-1]

            inputs = {"input_ids": input_ids,
                      "attention_mask": input_mask,
                      "token_type_ids": segment_ids,

                      "start_id": start_id,
                      "entity_ids": ent_ids,
                      "entity_position_ids": ent_pos_ids,
                      "entity_segment_ids": ent_seg_ids,
                      "entity_attention_mask": ent_mask,

                      "k_ent_ids": k_ent_ids,
                      "k_label": k_label,

                      "k_des_ids": des_input_ids,
                      "k_des_mask_one": des_att_mask_one,
                      "k_des_seg": des_segment_one,
                      "k_des_mask": des_mask,
                      "labels": None,
                      }

            logits = model(**inputs)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

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
    if args.task_name in ['openentity', 'figer']:
        from KFormers_roberta_distilbert_modeling import KFormersForEntityTyping as KFormersDownstreamModel
    elif args.task_name in ['tacred', 'fewrel']:
        from KFormers_roberta_distilbert_modeling import KFormersForRelationClassification as KFormersDownstreamModel
    elif args.task_name in ['sst2', 'eem']:
        from KFormers_roberta_distilbert_modeling import KFormersForSequenceClassification as KFormersDownstreamModel
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
    k_config_class, k_model_class, k_tokenizer_class = KNOWLEDGE_MODEL_CLASSES[args.knowledge_model_type]

    backbone_tokenizer = backbone_tokenizer_class.from_pretrained(args.backbone_model_name_or_path)
    knowledge_tokenizer = k_tokenizer_class.from_pretrained(args.knowledge_model_name_or_path)

    args.tokenizer = backbone_tokenizer

    # get num_label
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    args.output_mode = output_modes[args.task_name]

    args, model = load_kformers(args, k_config_class, k_model_class, KFormersDownstreamModel)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logging.info('loading backbone model: {}, knowledge module model: {}'.format(args.backbone_model_type,
                                                                                 args.knowledge_model_type))
    model.to(device)

    print('backbone_seq_length=', args.backbone_seq_length, ', knowledge_seq_length=', args.knowledge_seq_length,
          ', max_ent_num=', args.max_ent_num, ', max_des_num=', args.max_des_num, ', train_batch_size=', args.train_batch_size,
          ', learning_rate=', args.learning_rate, ', alpha, beta=', args.alpha, args.beta, ', seed=', args.seed)

    logging.info('backbone_seq_length={}, knowledge_seq_length={}, max_ent_num={}, max_des_num={}, '
                 'train_batch_size={}, learning_rate={}, alpha, beta={}, {}, seed={}'.format(args.backbone_seq_length,
                 args.knowledge_seq_length, args.max_ent_num, args.max_des_num,
                 args.train_batch_size, args.learning_rate, args.alpha, args.beta, args.seed))
    # ## Training
    if args.mode == 'train':
        from data_utils import load_and_cache_examples
        if args.task_name in ['sst2', 'eem']:
            # there is no test data in sst2(glue) and eem
            test_dataset = load_and_cache_examples(args, processor, backbone_tokenizer, knowledge_tokenizer,
                                                   dataset_type='dev', evaluate=True)
        else:
            test_dataset = load_and_cache_examples(args, processor, backbone_tokenizer, knowledge_tokenizer,
                                                   dataset_type='test', evaluate=True)
        val_dataset = load_and_cache_examples(args, processor, backbone_tokenizer, knowledge_tokenizer,
                                              dataset_type='dev', evaluate=True)
        train_dataset = load_and_cache_examples(args, processor, backbone_tokenizer, knowledge_tokenizer,
                                                dataset_type='train', evaluate=False)
        do_train(args, model, train_dataset, val_dataset, test_dataset)


if __name__ == "__main__":
    main()