import time
import logging
import torch
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

from transformers.optimization import AdamW, get_linear_schedule_with_warmup


from parameters import parse_args
from utils import compute_metrics
from data_utils import output_modes, processors, final_metric

logger = logging.getLogger(__name__)

BACKBONE_MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizerFast),
}
KNOWLEDGE_MODEL_CLASSES = {
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizerFast),
}


def setup(rank, world_size):
    # initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size, )
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def load_kformers(args, backbone_config_class, k_config_class, backbone_model_class, k_model_class,
                  KFormersDownstreamModel, BaselineDownstreamModel):

    config_backbone = backbone_config_class.from_pretrained(args.backbone_model_name_or_path, output_hidden_states=True)
    config_backbone.num_labels = args.num_labels

    if args.add_knowledge:
        config_k = k_config_class.from_pretrained(args.knowledge_model_name_or_path,
                                                  output_hidden_states=True)  # 这是DistillBert的config
        kformers_model = KFormersDownstreamModel(config=config_backbone, config_k=config_k,
                                                 backbone_knowledge_dict=args.backbone_knowledge_dict)
        # Load or initialize parameters.
        load_parameters = "pretrained"
        if load_parameters == "initialization":
            # Initialize with normal distribution.
            for n, p in list(kformers_model.named_parameters()):
                if 'gamma' not in n and 'beta' not in n:
                    p.data.normal_(0, 0.02)
        else:
            # Initialize with pretrained model.
            state_dict = kformers_model.state_dict()  # KFormers的全部参数

            # for x in state_dict.keys():
            #     print(x)

            backbone_model_dict = backbone_model_class.from_pretrained(args.backbone_model_name_or_path,
                                                                       config=config_backbone).state_dict()
            knowledge_model_dict = k_model_class.from_pretrained(args.knowledge_model_name_or_path,
                                                                 config=config_k).state_dict()

            # for k, v in backbone_model_dict.items():
            #     print(k)

            # for k, v in knowledge_model_dict.items():
            #     print(k)

            news_kformers_state_dict = OrderedDict()
            for key, value in backbone_model_dict.items():
                key = key.replace(args.backbone_model_type, 'kformers')
                if 'embeddings' in key:
                    news_kformers_state_dict[key] = value
                # "classifier.out_proj.weight", "classifier.out_proj.bias"这是roberta的
                elif key in ["kformers.pooler.dense.weight", "kformers.pooler.dense.bias",
                             "classifier.weight", "classifier.bias", "classifier.out_proj.weight",
                             "classifier.out_proj.bias"]:  # from_pretrained有的
                    # news_kformers_state_dict[key] = value
                    continue
                elif key in ["classifier.dense.weight", "classifier.dense.bias"]:  # for roberta
                    if args.task_name in ['tacred', 'fewrel']:
                        # if the task is relation classification, the output layer is not initialized by checkpoints
                        pass
                    else:
                        news_kformers_state_dict[key] = value
                elif key in ["classifier.weight", "classifier.bias"]:  # from_pretrained有的
                    pass
                elif key in ["lm_head.bias", "lm_head.dense.weight", "lm_head.dense.bias",
                             "lm_head.layer_norm.weight", "lm_head.layer_norm.bias",
                             "lm_head.decoder.weight"]:  # load有的参数
                    pass
                else:
                    i = key.split('kformers.encoder.layer.')[-1]
                    i = i.split('.')[0]
                    i = int(i)
                    key = key.replace('kformers.encoder.layer.%s' % i, 'kformers.encoder.layer.%s.backbone_layer' % i)
                    news_kformers_state_dict[key] = value

            backbone_knowledge_dict_reverse = {v: k for k, v in args.backbone_knowledge_dict.items()}
            for key, value in knowledge_model_dict.items():
                key = key.replace(args.knowledge_model_type, 'kformers')
                if 'embeddings' in key:
                    key = key.replace('kformers.embeddings', 'kformers.k_embeddings')
                    news_kformers_state_dict[key] = value
                elif key in ["pre_classifier.weight", "pre_classifier.bias", "classifier.weight",
                             "classifier.bias"]:  # from_pre
                    pass
                elif key in ["vocab_transform.weight", "vocab_transform.bias", "vocab_layer_norm.weight",
                             "vocab_layer_norm.bias", "vocab_projector.weight",
                             "vocab_projector.bias"]:  # 这是load checkpoint有的参数
                    pass
                else:
                    j = key.split('kformers.transformer.layer.')[-1]
                    j = j.split('.')[0]
                    i = backbone_knowledge_dict_reverse[int(j)]
                    key = key.replace('kformers.transformer.layer.%s' % j, 'kformers.encoder.layer.%s.k_layer' % i)
                    news_kformers_state_dict[key] = value

            # for k, v in news_kformers_state_dict.items():
            #     # print(k, v.size())
            #     print(k)
            # 在这里设置K-module的参数更新不更新
            if not args.update_K_module:
                for k, v in news_kformers_state_dict.items():
                    if 'k_' in k:
                        v.requires_grad = False
            state_dict.update(news_kformers_state_dict)
            kformers_model.load_state_dict(state_dict=state_dict)
        return kformers_model

    else:
        baseline_model = BaselineDownstreamModel.from_pretrained(args.backbone_model_name_or_path, config=config_backbone)

        return baseline_model


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

            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                input_ids, attention_mask, token_type_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                k_input_ids, k_mask, k_attention_mask, k_token_type_ids = batch[-5], batch[-4], batch[-3], batch[-2]
                labels = batch[-1]
            else:
                input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
                k_input_ids, k_mask, k_attention_mask, k_token_type_ids = batch[-5], batch[-4], batch[-3], batch[-2]
                labels = batch[-1]

            if args.add_knowledge:
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                          "k_input_ids_list": k_input_ids,
                          "k_mask": k_mask,
                          "k_attention_mask_list": k_attention_mask,
                          "k_token_type_ids_list": k_token_type_ids if args.knowledge_model_type in ['distilbert'] else None,
                          "labels": labels,
                          }
            else:
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                          "k_input_ids_list": None,
                          "k_mask": None,
                          "k_attention_mask_list": None,
                          "k_token_type_ids_list": None,
                          "labels": labels,
                          }

            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                inputs['start_id'] = start_id

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
                test_results = do_eval(model, args, test_dataset, global_step)
                t = eval_results[final_metric[args.task_name]]
                if t > best_dev_result:  # f1
                    best_dev_result = eval_results[final_metric[args.task_name]]
                    logger.info('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                    print('epoch: {}, global step: {}, dev results: {}**'.format(epoch, global_step, eval_results))
                    logger.info('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
                    print('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
                else:
                    logger.info('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
                    print('epoch: {}, global step: {}, dev results: {}'.format(epoch, global_step, eval_results))
                    logger.info('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
                    print('epoch: {}, global step: {}, test results: {}'.format(epoch, global_step, test_results))
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        # evaluate per epoch
        if args.local_rank in [-1, 0]:
            eval_results = do_eval(model, args, val_dataset, global_step=epoch)
            test_results = do_eval(model, args, test_dataset, global_step=epoch)
            t = eval_results[final_metric[args.task_name]]
            if t > best_dev_result:  # f1
                best_dev_result = eval_results[final_metric[args.task_name]]
                logger.info('epoch: {},  dev results: {}**'.format(epoch, eval_results))
                print('epoch: {}, dev results: {}**'.format(epoch, eval_results))
                logger.info('epoch: {}, test results: {}'.format(epoch, test_results))
                print('epoch: {}, test results: {}'.format(epoch, test_results))
            else:
                logger.info('epoch: {}, dev results: {}'.format(epoch, eval_results))
                print('epoch: {}, dev results: {}'.format(epoch, eval_results))
                logger.info('epoch: {}, test results: {}'.format(epoch, test_results))
                print('epoch: {}, test results: {}'.format(epoch, test_results))

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
    logging.info("train time:{}".format(time.time() - start_time))


def do_eval(model, args, val_dataset, global_step):
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=args.valid_batch_size)

    preds = None
    out_label_ids = None
    eval_iterator = tqdm(val_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(eval_iterator):
        # if step > 3:
        #     break

        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():

            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                input_ids, attention_mask, token_type_ids, start_id = batch[0], batch[1], batch[2], batch[3]
                k_input_ids, k_mask, k_attention_mask, k_token_type_ids = batch[-5], batch[-4], batch[-3], batch[-2]
                labels = batch[-1]
            else:
                input_ids, attention_mask, token_type_ids = batch[0], batch[1], batch[2]
                k_input_ids, k_mask, k_attention_mask, k_token_type_ids = batch[-5], batch[-4], batch[-3], batch[-2]
                labels = batch[-1]

            if args.add_knowledge:
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                          "k_input_ids_list": k_input_ids,
                          "k_mask": k_mask,
                          "k_attention_mask_list": k_attention_mask,
                          "k_token_type_ids_list": k_token_type_ids if args.knowledge_model_type in [
                              'distilbert'] else None,
                          "labels": None,
                          }
            else:
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids if args.backbone_model_type in ['bert', 'unilm'] else None,
                          "k_input_ids_list": None,
                          "k_mask": None,
                          "k_attention_mask_list": None,
                          "k_token_type_ids_list": None,
                          "labels": None,
                          }

            if args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
                inputs['start_id'] = start_id

            logits = model(**inputs)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    if args.task_name in ['open_entity',  'figer']:
        pass
    elif args.task_name in ['tacred', 'fewrel']:
        preds = np.argmax(preds, axis=1)

    result = compute_metrics(args.task_name, preds, out_label_ids)
    return result


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    args = parse_args()
    if args.task_name in ['open_entity', 'figer']:
        from KFormers_roberta_distilbert_modeling import KFormersForEntityTyping as KFormersDownstreamModel
        from KFormers_roberta_distilbert_modeling import RobertaForEntityTyping as BaselineDownstreamModel
    elif args.task_name in ['tacred', 'fewrel']:
        from KFormers_roberta_distilbert_modeling import KFormersForRelationClassification as KFormersDownstreamModel
        from KFormers_roberta_distilbert_modeling import RobertaForRelationClassification as BaselineDownstreamModel
    elif args.task_name in ['sst2', 'eem']:
        from KFormers_roberta_distilbert_modeling import RobertaForSequenceClassification as BaselineDownstreamModel
        from KFormers_roberta_distilbert_modeling import KFormersForSequenceClassification as KFormersDownstreamModel
    else:
        KFormersDownstreamModel = None
        BaselineDownstreamModel = None

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

    # get num_label
    if args.task_name == 'tacred':
        processor = processors[args.task_name](tokenizer=backbone_tokenizer, k_tokenizer=knowledge_tokenizer, negative_sample=args.negative_sample)
    else:
        processor = processors[args.task_name](tokenizer=backbone_tokenizer, k_tokenizer=knowledge_tokenizer)
    # processor = processors[args.task_name](tokenizer=backbone_tokenizer, k_tokenizer=knowledge_tokenizer)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    args.output_mode = output_modes[args.task_name]

    model = load_kformers(args, backbone_config_class, k_config_class, backbone_model_class, k_model_class,
                          KFormersDownstreamModel, BaselineDownstreamModel)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    logging.info('loading backbone model: {}, knowledge module model: {}'.format(args.backbone_model_type,
                                                                                 args.knowledge_model_type))
    model.to(device)

    logger.info("Training/evaluation parameters %s", args)
    print("Training/evaluation parameters %s", args)
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