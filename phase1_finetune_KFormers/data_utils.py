import csv
import numpy as np
import sys
from io import open
import random
import json
import logging
import os
import torch
from typing import List, TextIO, Dict
from collections import Counter, OrderedDict, defaultdict, namedtuple
from tqdm import tqdm
from multiprocessing.pool import Pool
from wikipedia2vec.dump_db import DumpDB
from contextlib import closing
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


HEAD_TOKEN = "[HEAD]"
TAIL_TOKEN = "[TAIL]"


def load_and_cache_examples(args, processor, tokenizer, k_tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    input_mode = input_modes[args.task_name]
    output_mode = output_modes[args.task_name]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}_ent={}_des={}_{}'.format(
        args.model_type, args.backbone_model_type, args.knowledge_model_type,
        args.task_name,
        dataset_type,
        str(args.backbone_seq_length),
        str(args.max_ent_num),
        str(args.max_des_num),
        str(args.knowledge_seq_length)))
    if os.path.exists(cached_features_file):
        logger.warning("===> Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.warning("===> Creating features from dataset file at {}, {}".format(str(args.data_dir), dataset_type))
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, dataset_type)
        else:
            examples = processor.get_train_examples(args.data_dir, dataset_type)
        if input_mode == 'sentence_pair':
            features = convert_examples_to_features_sentence_pair(examples,
                                                                  args.backbone_seq_length, args.knowledge_seq_length, args.max_num_entity,
                                                                  tokenizer, k_tokenizer)
        elif input_mode == 'single_sentence':
            features = convert_examples_to_features_single_sentence(args, examples, args.entity_vocab,
                                                                    args.backbone_seq_length, args.knowledge_seq_length,
                                                                    args.max_ent_num, args.max_des_num, 2,
                                                                    tokenizer, k_tokenizer,)
        elif input_mode == 'entity_sentence':
            features = convert_examples_to_features_entity_typing(args, examples, args.entity_vocab,
                                                                  args.backbone_seq_length, args.knowledge_seq_length,
                                                                  args.max_ent_num, args.max_des_num, 2,
                                                                  tokenizer, k_tokenizer,
                                                                  )
        elif input_mode == "entity_entity_sentence":
            features = convert_examples_to_features_relation_classification(args, examples, args.entity_vocab,
                                                                            args.backbone_seq_length, args.knowledge_seq_length,
                                                                            args.max_ent_num, args.max_des_num, 30,
                                                                            tokenizer, k_tokenizer,)
        else:
            features = None
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.warning("===> Saving features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # if isinstance(features, list):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    ent_ids = torch.tensor([f.ent_ids for f in features], dtype=torch.long)
    ent_mask = torch.tensor([f.ent_mask for f in features], dtype=torch.long)
    ent_seg_ids = torch.tensor([f.ent_seg_ids for f in features], dtype=torch.long)
    ent_pos_ids = torch.tensor([f.ent_pos_ids for f in features], dtype=torch.long)

    k_ent_ids = torch.tensor([f.k_ent_ids for f in features], dtype=torch.long)
    k_label = torch.tensor([f.k_label for f in features], dtype=torch.long)

    des_input_ids = torch.tensor([f.des_input_ids for f in features], dtype=torch.long)
    des_att_mask_one = torch.tensor([f.des_att_mask_one for f in features], dtype=torch.long)
    des_segment_one = torch.tensor([f.des_segment_one for f in features], dtype=torch.long)
    des_mask = torch.tensor([f.des_mask for f in features], dtype=torch.long)


    if args.task_name in ['openentity', 'figer']:
        label_id = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        label_id = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if args.task_name in ['fewrel', 'tacred', 'openentity', 'figer']:

        start_id = torch.tensor([f.start_id for f in features], dtype=torch.float)
        dataset = TensorDataset(input_ids, input_mask, segment_ids, start_id,
                                ent_ids, ent_mask, ent_seg_ids, ent_pos_ids,
                                k_ent_ids, k_label,
                                des_input_ids, des_att_mask_one, des_segment_one, des_mask,
                                label_id)
    else:
        dataset = TensorDataset(input_ids, input_mask, segment_ids,
                                ent_ids, ent_mask, ent_seg_ids, ent_pos_ids,
                                k_ent_ids, k_label,
                                des_input_ids, des_att_mask_one, des_segment_one, des_mask,
                                label_id)

    return dataset


def _tensorize_batch(inputs, padding_value, max_length, neighbour=False):
    if neighbour:
        inputs_padded = []
        for x in inputs:
            q_des_pad = []
            for y in x:
                padding = [padding_value] * (max_length - len(y))
                y += padding
                q_des_pad.append(y)
            t1 = torch.tensor(q_des_pad)
            # print(t1.size(), x)
            inputs_padded.append(t1.unsqueeze(0))
        t2 = torch.cat(inputs_padded, dim=0)
        return t2
    else:
        inputs_padded = []
        for x in inputs:
            padding = [padding_value] * (max_length - len(x))
            x += padding
            t1 = torch.tensor(x)
            inputs_padded.append(t1.unsqueeze(0))
        t2 = torch.cat(inputs_padded, dim=0)
        return t2


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_sentence_pair(examples, origin_seq_length, knowledge_seq_length, max_num_entity,
                                               tokenizer, k_tokenizer,
                                               pad_on_left=False,
                                               mask_padding_with_zero=True):

    max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=True)  # 32 - 3
    k_max_num_tokens = knowledge_seq_length - k_tokenizer.num_special_tokens_to_add(pair=False)  # 32 - 2
    # 1. tokenize: input & knowledge
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))

        query, key, neighbours, label = example.text_a, example.text_b, example.neighbour, example.label

        query = tokenizer.tokenize(query)
        key = tokenizer.tokenize(key)
        query = tokenizer.convert_tokens_to_ids(query)
        key = tokenizer.convert_tokens_to_ids(key)
        _truncate_seq_pair(query, key, max_length=max_num_tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=query, token_ids_1=key)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)  # roberta 不使用

        padding_length = origin_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([tokenizer.pad_token_id] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([tokenizer.pad_token_type_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        if ex_index < 5:
            logger.info("*** ===> Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label))
        # ==== backbone ====
        # ==== knowledge ====
        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        neighbours_mask = [1] * min(max_num_entity, len(neighbours)) + [0] * (max_num_entity - min(max_num_entity, len(neighbours)))  # 这是entity的mask不是description
        neighbours = neighbours[: max_num_entity] + ["PAD DESCRIPTION"] * (max_num_entity - len(neighbours))
        for x in neighbours:
            x = k_tokenizer.tokenize(x)
            x = x[: k_max_num_tokens]
            x = k_tokenizer.convert_tokens_to_ids(x)
            x = k_tokenizer.build_inputs_with_special_tokens(token_ids_0=x)  # input_ids_knowledge
            x_2 = [1 if mask_padding_with_zero else 0] * len(x)  # input_mask
            x_3 = [0] * len(x)  # segment_id
            padding_length = knowledge_seq_length - len(x)
            if pad_on_left:
                x = ([k_tokenizer.pad_token_id] * padding_length) + x
                x_2 = ([0 if mask_padding_with_zero else 1] * padding_length) + x_2
                x_3 = ([k_tokenizer.pad_token_type_id] * padding_length) + x_3
            else:
                x = x + ([k_tokenizer.pad_token_id] * padding_length)
                x_2 = x_2 + ([0 if mask_padding_with_zero else 1] * padding_length)
                x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length

            neighbour_one.append(x)
            neighbour_att_mask_one.append(x_2)
            neighbour_segment_one.append(x_3)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens_k: %s" % " ".join([str(x_) for x_ in neighbours]))
            logger.info("neighbours_mask: %s" % " ".join([str(x_) for x_ in neighbours_mask]))
            logger.info("input_ids_k: %s" % " ".join([str(x) for x in neighbour_one]))
            logger.info("input_mask_k: %s" % " ".join([str(x) for x in neighbour_att_mask_one]))
            logger.info("segment_ids_k: %s" % " ".join([str(x) for x in neighbour_segment_one]))
        # ==== knowledge ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label,
                          k_input_ids=neighbour_one,
                          k_mask=neighbours_mask,  # 表示有几条有效description
                          k_input_mask=neighbour_att_mask_one,
                          k_segment_ids=neighbour_segment_one,  # distilBert没有使用token_type_embedding
                          ))


    return features


def convert_examples_to_features_single_sentence(args, examples, entity_vocab,
                                                 origin_seq_length, knowledge_seq_length,
                                                 max_ent_num, max_des_num, max_mention_length,
                                                tokenizer, k_tokenizer,
                                               ):

    _, QID_description_dict = load_description(args.qid_file)
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        text_a, label = example.text_a, example.label
        tokens = tokenizer.tokenize(text_a)
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # pad
        padding_length = origin_seq_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        # entity identifier
        entity_identifier_ids = [0, 0]  # 单句分类这里如何处理？
        entity_identifier_segment_ids = [0, 0]
        entity_identifier_attention_mask = [0, 0]
        start, end = 0, 1
        entity_identifier_position_ids = list(range(start, end))[:max_mention_length]
        entity_identifier_position_ids += [-1] * (max_mention_length - len(entity_identifier_position_ids))
        # 处理knowledge：entity
        k_ent_ids = [entity_vocab[ent[0]] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_ids = k_ent_ids[: max_ent_num]
        k_ent_scores = [ent[-1] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_scores = k_ent_scores[: max_ent_num]
        if len(k_ent_ids) == 0:
            k_ent_ids = [entity_vocab[MASK_TOKEN]]
            k_ent_scores = [5]
        k_ent_ids += random.sample(list(range(4, len(entity_vocab))), (max_ent_num - len(k_ent_ids)))
        k_ent_scores += [-1] * (max_ent_num - len(k_ent_scores))
        tmp = list(zip(k_ent_ids, k_ent_scores))
        random.shuffle(tmp)
        k_ent_ids, k_ent_scores = zip(*tmp)
        k_ent_ids, k_ent_scores = list(k_ent_ids), list(k_ent_scores)
        k_label = k_ent_scores.index(max(k_ent_scores))

        # 处理knowledge：description
        des_input_ids, des_att_mask_one, des_segment_one = [], [], []

        k_ent_qids = [ent[0] for ent in example.entities]
        k_des = [QID_description_dict[qid] for qid in k_ent_qids if qid in QID_description_dict]
        k_des = k_des[: max_des_num]

        des_mask = [1] * len(k_des) + [0] * (max_des_num - len(k_des))
        k_des = k_des[: max_des_num] + [k_tokenizer.pad_token] * (max_des_num - len(k_des))
        for x in k_des:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1] * len(x_1)  # input_mask
            x_3 = [0] * len(x)  # segment
            # pad
            padding_length = knowledge_seq_length - len(x)
            x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
            x_2 = x_2 + ([0] * padding_length)
            x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            des_input_ids.append(x_1)
            des_att_mask_one.append(x_2)
            des_segment_one.append(x_3)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label))

            logger.info("ent_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_ids]))
            logger.info("ent_mask: %s" % " ".join([str(x_) for x_ in entity_identifier_attention_mask]))
            logger.info("ent_seg_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_segment_ids]))
            logger.info("ent_pos_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_position_ids]))

            logger.info("k_ent_ids: %s" % " ".join([str(x_) for x_ in k_ent_ids]))
            logger.info("k_label: %s" % k_label)

            logger.info("des_input_ids: %s" % " ".join([str(x_) for x_ in des_input_ids]))
            logger.info("des_att_mask_one: %s" % " ".join([str(x) for x in des_att_mask_one]))
            logger.info("des_segment_one: %s" % " ".join([str(x) for x in des_segment_one]))
            logger.info("des_mask: %s" % " ".join([str(x) for x in des_mask]))
        # ==== knowledge ====
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label,

                          ent_ids=entity_identifier_ids,
                          ent_mask=entity_identifier_attention_mask,
                          ent_seg_ids=entity_identifier_segment_ids,
                          ent_pos_ids=entity_identifier_position_ids,

                          k_ent_ids=k_ent_ids,
                          k_label=k_label,

                          des_input_ids=des_input_ids,
                          des_att_mask_one=des_att_mask_one,
                          des_segment_one=des_segment_one,
                          des_mask=des_mask,  # 表示有几条有效description
                          ))

    return features


def convert_examples_to_features_relation_classification(args, examples, entity_vocab,
                                                         origin_seq_length, knowledge_seq_length,
                                                         max_ent_num, max_des_num, max_mention_length,
                                                         tokenizer, k_tokenizer,):

    _, QID_description_dict = load_description(args.qid_file)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        text_a = example.text_a
        start_0, end_0, start_1, end_1 = example.text_b
        # sub, and then obj (有空格啊，妈的)
        before_sub = text_a[:start_0].strip()
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(before_sub)
        sub_start = len(tokens)
        tokens += [HEAD_TOKEN]
        sub = text_a[start_0: end_0 + 1].strip()
        tokens += tokenizer.tokenize(sub)
        tokens += [HEAD_TOKEN]
        sub_end = len(tokens)
        between_sub_obj = text_a[end_0 + 1: start_1].strip()
        tokens += tokenizer.tokenize(between_sub_obj)
        obj_start = len(tokens)
        tokens += [TAIL_TOKEN]
        obj = text_a[start_1: end_1 + 1].strip()
        tokens += tokenizer.tokenize(obj)
        tokens += [TAIL_TOKEN]
        obj_end = len(tokens)
        after_obj = text_a[end_1 + 1:].strip()
        tokens += tokenizer.tokenize(after_obj) + [tokenizer.sep_token]

        tokens = tokens[: origin_seq_length]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # pad
        padding_length = origin_seq_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        # label
        label_id = example.label
        # sure that sub & obj are included in the sequence
        if sub_start > origin_seq_length - 1:
            sub_start = 0
        if obj_start > origin_seq_length - 1:
            obj_start = 0
        if sub_end > origin_seq_length - 1:
            sub_end = origin_seq_length
        if obj_end > origin_seq_length:
            obj_end = origin_seq_length
        # the sub_special_start_id is an array, where the idx of start id is 1, other position is 0.
        subj_special_start_id = np.zeros(origin_seq_length)
        obj_special_start_id = np.zeros(origin_seq_length)
        subj_special_start_id[sub_start] = 1
        obj_special_start_id[obj_start] = 1
        # entity identifier
        entity_identifier_ids = [1, 2]
        entity_identifier_segment_ids = [0, 0]
        entity_identifier_attention_mask = [1, 1]
        entity_identifier_position_ids = []
        for span_name in ([sub_start, sub_end], [obj_start, obj_end]):
            start, end = span_name
            position_ids = list(range(start, end))[:max_mention_length]
            position_ids += [-1] * (max_mention_length - len(position_ids))
            assert len(position_ids) == max_mention_length
            entity_identifier_position_ids.append(position_ids)
        # 处理knowledge：entity
        k_ent_ids = [entity_vocab[ent[0]] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_ids = k_ent_ids[: max_ent_num]
        k_ent_scores = [ent[-1] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_scores = k_ent_scores[: max_ent_num]
        if len(k_ent_ids) == 0:
            k_ent_ids = [entity_vocab[MASK_TOKEN]]
            k_ent_scores = [5]
        k_ent_ids += random.sample(list(range(4, len(entity_vocab))), (max_ent_num - len(k_ent_ids)))
        k_ent_scores += [-1] * (max_ent_num - len(k_ent_scores))
        tmp = list(zip(k_ent_ids, k_ent_scores))
        random.shuffle(tmp)
        k_ent_ids, k_ent_scores = zip(*tmp)
        k_ent_ids, k_ent_scores = list(k_ent_ids), list(k_ent_scores)
        k_label = k_ent_scores.index(max(k_ent_scores))
        # 处理knowledge：description
        des_input_ids, des_att_mask_one, des_segment_one = [], [], []
        k_ent_qids = [ent[0] for ent in example.entities]
        k_des = [QID_description_dict[qid] for qid in k_ent_qids if qid in QID_description_dict]
        k_des = k_des[: max_des_num]

        des_mask = [1] * len(k_des) + [0] * (max_des_num - len(k_des))
        k_des = k_des[: max_des_num] + [k_tokenizer.pad_token] * (max_des_num - len(k_des))
        for x in k_des:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1] * len(x_1)  # input_mask
            x_3 = [0] * len(x)  # segment
            # pad
            padding_length = knowledge_seq_length - len(x)
            x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
            x_2 = x_2 + ([0] * padding_length)
            x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            des_input_ids.append(x_1)
            des_att_mask_one.append(x_2)
            des_segment_one.append(x_3)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
            logger.info("start_id: {}".format((sub_start, obj_start)))

            logger.info("ent_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_ids]))
            logger.info("ent_mask: %s" % " ".join([str(x_) for x_ in entity_identifier_attention_mask]))
            logger.info("ent_seg_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_segment_ids]))
            logger.info("ent_pos_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_position_ids]))

            logger.info("k_ent_ids: %s" % " ".join([str(x_) for x_ in k_ent_ids]))
            logger.info("k_label: %s" % k_label)

            logger.info("des_input_ids: %s" % " ".join([str(x_) for x_ in des_input_ids]))
            logger.info("des_att_mask_one: %s" % " ".join([str(x) for x in des_att_mask_one]))
            logger.info("des_segment_one: %s" % " ".join([str(x) for x in des_segment_one]))
            logger.info("des_mask: %s" % " ".join([str(x) for x in des_mask]))
        # ==== knowledge ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=example.label,
                          start_id=(subj_special_start_id, obj_special_start_id),

                          ent_ids=entity_identifier_ids,
                          ent_mask=entity_identifier_attention_mask,
                          ent_seg_ids=entity_identifier_segment_ids,
                          ent_pos_ids=entity_identifier_position_ids,

                          k_ent_ids=k_ent_ids,
                          k_label=k_label,

                          des_input_ids=des_input_ids,
                          des_att_mask_one=des_att_mask_one,
                          des_segment_one=des_segment_one,
                          des_mask=des_mask,  # 表示有几条有效description
                          ))

    return features


def convert_examples_to_features_entity_typing(args, examples, entity_vocab,
                                               origin_seq_length, knowledge_seq_length, max_ent_num, max_des_num, max_mention_length,
                                               tokenizer, k_tokenizer,
                                               ):

    _, QID_description_dict = load_description(args.qid_file)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        start, end = example.text_b[0], example.text_b[1]
        sentence = example.text_a
        tokens_0_start = tokenizer.tokenize(sentence[:start])
        tokens_start_end = tokenizer.tokenize(sentence[start:end])
        tokens_end_last = tokenizer.tokenize(sentence[end:])
        tokens = [tokenizer.cls_token] + tokens_0_start + tokenizer.tokenize("[ENTITY]") + tokens_start_end + tokenizer.tokenize("[ENTITY]") + tokens_end_last + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]
        start = 1 + len(tokens_0_start)
        end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = origin_seq_length - len(input_ids)
        # pad
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        # label
        label_id = example.label
        start_id = np.zeros(origin_seq_length)
        if start >= origin_seq_length:
            start = 0  # 如果entity被截断了，就使用CLS位代替
        start_id[start] = 1

        # entity identifier
        entity_identifier_ids = [1, 0]
        entity_identifier_attention_mask = [1, 0]
        entity_identifier_segment_ids = [0, 0]
        entity_identifier_position_ids = list(range(start, end))[:max_mention_length]
        entity_identifier_position_ids += [-1] * (max_mention_length - end + start)
        entity_identifier_position_ids = [entity_identifier_position_ids, [-1] * max_mention_length]  # ？
        # 处理knowledge：entity
        k_ent_ids = [entity_vocab[ent[0]] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_ids = k_ent_ids[: max_ent_num]
        k_ent_scores = [ent[-1] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_scores = k_ent_scores[: max_ent_num]
        if len(k_ent_ids) == 0:
            k_ent_ids = [entity_vocab[MASK_TOKEN]]
            k_ent_scores = [5]
        k_ent_ids += random.sample(list(range(4, len(entity_vocab))), (max_ent_num - len(k_ent_ids)))
        k_ent_scores += [-1] * (max_ent_num - len(k_ent_scores))
        tmp = list(zip(k_ent_ids, k_ent_scores))
        random.shuffle(tmp)
        k_ent_ids, k_ent_scores = zip(*tmp)
        k_ent_ids, k_ent_scores = list(k_ent_ids), list(k_ent_scores)
        k_label = k_ent_scores.index(max(k_ent_scores))
        # 处理knowledge：description
        des_input_ids, des_att_mask_one, des_segment_one = [], [], []

        k_ent_qids = [ent[0] for ent in example.entities]
        k_des = [QID_description_dict[qid] for qid in k_ent_qids if qid in QID_description_dict]
        k_des = k_des[: max_des_num]

        des_mask = [1] * len(k_des) + [0] * (max_des_num - len(k_des))
        k_des = k_des[: max_des_num] + [k_tokenizer.pad_token] * (max_des_num - len(k_des))
        for x in k_des:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1] * len(x_1)  # input_mask
            x_3 = [0] * len(x)  # segment
            # pad
            padding_length = knowledge_seq_length - len(x)
            x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
            x_2 = x_2 + ([0] * padding_length)
            x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            des_input_ids.append(x_1)
            des_att_mask_one.append(x_2)
            des_segment_one.append(x_3)
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
            logger.info("start_id: {}".format(start_id))

            logger.info("ent_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_ids]))
            logger.info("ent_mask: %s" % " ".join([str(x_) for x_ in entity_identifier_attention_mask]))
            logger.info("ent_seg_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_segment_ids]))
            logger.info("ent_pos_ids: %s" % " ".join([str(x_) for x_ in entity_identifier_position_ids]))

            logger.info("k_ent_ids: %s" % " ".join([str(x_) for x_ in k_ent_ids]))
            logger.info("k_label: %s" % k_label)

            logger.info("des_input_ids: %s" % " ".join([str(x_) for x_ in des_input_ids]))
            logger.info("des_att_mask_one: %s" % " ".join([str(x) for x in des_att_mask_one]))
            logger.info("des_segment_one: %s" % " ".join([str(x) for x in des_segment_one]))
            logger.info("des_mask: %s" % " ".join([str(x) for x in des_mask]))
        # ==== knowledge ====
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          start_id=start_id,

                          ent_ids=entity_identifier_ids,
                          ent_mask=entity_identifier_attention_mask,
                          ent_seg_ids=entity_identifier_segment_ids,
                          ent_pos_ids=entity_identifier_position_ids,

                          k_ent_ids=k_ent_ids,
                          k_label=k_label,

                          des_input_ids=des_input_ids,
                          des_att_mask_one=des_att_mask_one,
                          des_segment_one=des_segment_one,
                          des_mask=des_mask, # 表示有几条有效description
                          ))
    return features



class InputFeatures(object):
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None, start_id=None, label_id=None,
                 ent_ids=None, ent_mask=None, ent_seg_ids=None, ent_pos_ids=None,
                 k_ent_ids=None, k_label=None,
                 des_input_ids=None, des_att_mask_one=None, des_segment_one=None, des_mask=None):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_id = start_id
        self.label_id = label_id

        self.ent_ids = ent_ids
        self.ent_mask = ent_mask
        self.ent_seg_ids = ent_seg_ids
        self.ent_pos_ids = ent_pos_ids

        self.k_ent_ids = k_ent_ids
        self.k_label = k_label

        self.des_input_ids = des_input_ids
        self.des_att_mask_one = des_att_mask_one
        self.des_segment_one = des_segment_one
        self.des_mask = des_mask




class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, entities=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.entities = entities
        self.label = label



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)



class OpenentityProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        label_list = ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group']
        return label_list

    def _create_examples(self, lines):
        examples = []
        label_list = self.get_labels()
        label_set = set()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0 for item in range(len(label_list))]
            for item in line['labels']:
                label_set.add(item)
                label[label_list.index(item)] = 1
            entities = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, entities=entities, label=label))
        return examples



class FigerProcessor(DataProcessor):


    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        label_list = ["/person/artist", "/person", "/transportation", "/location/cemetery", "/language", "/location",
                      "/location/city", "/transportation/road", "/person/actor", "/person/soldier",
                      "/person/politician", "/location/country", "/geography", "/geography/island", "/people",
                      "/people/ethnicity", "/internet", "/internet/website", "/broadcast_network", "/organization",
                      "/organization/company", "/person/athlete", "/organization/sports_team", "/location/county",
                      "/geography/mountain", "/title", "/person/musician", "/event",
                      "/organization/educational_institution",
                      "/person/author", "/military", "/astral_body", "/written_work", "/event/military_conflict",
                      "/person/engineer",
                      "/event/attack", "/organization/sports_league", "/government", "/government/government",
                      "/location/province",
                      "/chemistry", "/music", "/education/educational_degree", "/education",
                      "/building/sports_facility",
                      "/building", "/government_agency", "/broadcast_program", "/living_thing", "/event/election",
                      "/location/body_of_water", "/person/director", "/park", "/event/sports_event", "/law",
                      "/product/ship", "/product", "/product/weapon", "/building/airport", "/software",
                      "/computer/programming_language",
                      "/computer", "/body_part", "/disease", "/art", "/art/film", "/person/monarch", "/game", "/food",
                      "/person/coach", "/government/political_party", "/news_agency", "/rail/railway", "/rail",
                      "/train",
                      "/play", "/god", "/product/airplane", "/event/natural_disaster", "/time", "/person/architect",
                      "/award", "/medicine/medical_treatment", "/medicine/drug", "/medicine",
                      "/organization/fraternity_sorority",
                      "/event/protest", "/product/computer", "/person/religious_leader", "/religion",
                      "/religion/religion",
                      "/building/theater", "/biology", "/livingthing", "/livingthing/animal", "/finance/currency",
                      "/finance",
                      "/organization/airline", "/product/instrument", "/location/bridge", "/building/restaurant",
                      "/medicine/symptom",
                      "/product/car", "/person/doctor", "/metropolitan_transit", "/metropolitan_transit/transit_line",
                      "/transit",
                      "/product/spacecraft", "/broadcast", "/broadcast/tv_channel", "/building/library",
                      "/education/department", "/building/hospital"]
        return label_list

    def _create_examples(self, lines):
        examples = []
        label_list = self.get_labels()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0] * len(label_list)
            for item in line['labels']:
                label[label_list.index(item)] = 1
            neighbour = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, entities=neighbour, label=label))
        return examples


TACRED_relations = ['per:siblings', 'per:parents', 'org:member_of', 'per:origin', 'per:alternate_names', 'per:date_of_death',
             'per:title', 'org:alternate_names', 'per:countries_of_residence', 'org:stateorprovince_of_headquarters',
             'per:city_of_death', 'per:schools_attended', 'per:employee_of', 'org:members', 'org:dissolved',
             'per:date_of_birth', 'org:number_of_employees/members', 'org:founded', 'org:founded_by',
             'org:political/religious_affiliation', 'org:website', 'org:top_members/employees', 'per:children',
             'per:cities_of_residence', 'per:cause_of_death', 'org:shareholders', 'per:age', 'per:religion',
             'NA',
             'org:parents', 'org:subsidiaries', 'per:country_of_birth', 'per:stateorprovince_of_death',
             'per:city_of_birth',
             'per:stateorprovinces_of_residence', 'org:country_of_headquarters', 'per:other_family',
             'per:stateorprovince_of_birth',
             'per:country_of_death', 'per:charges', 'org:city_of_headquarters', 'per:spouse']



class TACREDProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_labels(self, ):
        labels = set(TACRED_relations)
        if 'NA' in labels:
            labels.discard("NA")
            return ["NA"] + sorted(labels)
        else:
            return sorted(labels)

    def _create_examples(self, lines, ):
        examples = []
        label_set = self.get_labels()
        label_map = {l: i for i, l in enumerate(label_set)}
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['text']
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0

            sub = (line["ents"][0][1], line["ents"][0][2])
            obj = (line["ents"][1][1], line["ents"][1][2])

            text_b = (sub[0], sub[1], obj[0], obj[1])

            label = line['label']
            label = label_map[label]
            neighbour = line['ann']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, entities=neighbour, label=label))
        random.shuffle(examples)
        return examples



class FewrelProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type):
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "train")
        return examples

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "{}".format(dataset_type))

    def get_labels(self):
        labels = ['P22', 'P449', 'P137', 'P57', 'P750', 'P102', 'P127', 'P1346', 'P410', 'P156', 'P26', 'P674', 'P306', 'P931',
         'P1435', 'P495', 'P460', 'P1411', 'P1001', 'P6', 'P413', 'P178', 'P118', 'P276', 'P361', 'P710', 'P155',
         'P740', 'P31', 'P1303', 'P136', 'P974', 'P407', 'P40', 'P39', 'P175', 'P463', 'P527', 'P17', 'P101', 'P800',
         'P3373', 'P2094', 'P135', 'P58', 'P206', 'P1344', 'P27', 'P105', 'P25', 'P1408', 'P3450', 'P84', 'P991',
         'P1877', 'P106', 'P264', 'P355', 'P937', 'P400', 'P177', 'P140', 'P1923', 'P706', 'P123', 'P131', 'P159',
         'P641', 'P412', 'P403', 'P921', 'P176', 'P59', 'P466', 'P241', 'P150', 'P86', 'P4552', 'P551', 'P364']
        return labels

    def _create_examples(self, lines, dataset_type):
        examples = []
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = line['text']
            text_b = (line['ents'][0][1], line['ents'][0][2], line['ents'][1][1],  line['ents'][1][2])
            neighbour = line['ents']
            label = line['label']
            label = label_map[label]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, entities=neighbour, label=label))
        return examples



class EEMProcessor(DataProcessor):
    def __init__(self, tokenizer=None, k_tokenizer=None):
        self.tokenizer = tokenizer
        self.k_tokenizer = k_tokenizer

    def get_train_examples(self, data_dir, dataset_type):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "train")
        return examples

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        labels = {"0": 0, "1": 1}
        return labels

    def _create_examples(self, lines, dataset_type):
        label_map = self.get_labels()
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            text_a = line['query']
            text_b = line["keyword"]
            neighbour = [x[1] for x in line['ents'] if x[1] != '']
            label = label_map[line['label']]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
        return examples



class Sst2Processor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type):
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "train")
        return examples

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        labels = {"0": 0, "1": 1}
        return labels

    def _create_examples(self, lines, dataset_type):
        label_map = self.get_labels()
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            text_a = line['sent']
            neighbour = line['ents']

            label = label_map[line['label']]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, entities=neighbour, label=label))
        return examples


def load_description(file):
    # load entity description, e.g., wikidata or wikipedia
    QID_entityName_dict = {}
    QID_description_dict = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            qid, name, des = line.strip().split('\t')
            QID_entityName_dict[qid] = name
            if des == 'NULL' or des == 'None':
                continue
            QID_description_dict[qid] = des
    return QID_entityName_dict, QID_description_dict



processors = {
    "sst2": Sst2Processor,
    "eem": EEMProcessor,
    "openentity": OpenentityProcessor,
    "figer": FigerProcessor,
    "tacred": TACREDProcessor,
    "fewrel": FewrelProcessor,
}

output_modes = {
    "qqp": "classification",
    "qnli": "classification",
    "wnli": "classification",
    "sst2": "classification",
    "eem": "classification",
    "openentity": "classification",
    "figer": "classification",
    "tacred": "classification",
    "fewrel": "classification",
}

input_modes = {
    "qqp": "sentence_pair",
    "qnli": "sentence_pair",
    "wnli": "sentence_pair",
    "sst2": "single_sentence",
    "eem": "sentence_pair",
    "openentity": "entity_sentence",
    "figer": "entity_sentence",
    "tacred": "entity_entity_sentence",
    "fewrel": "entity_entity_sentence",
}

final_metric = {
    'sst2': 'accuracy',
    "eem": 'roc_auc',
    "openentity": 'micro_F1',
    "figer": 'micro_F1',
    "tacred": 'micro_F1',
    "fewrel": 'micro_F1'

}




Entity = namedtuple("Entity", ["title", "language"])
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
ENTITY_TOKEN = "[ENTITY]"
HEAD_TOKEN = "[HEAD]"
TAIL_TOKEN = "[TAIL]"

class EntityVocab(object):
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            self._parse_tsv_vocab_file(vocab_file)
        else:
            self._parse_jsonl_vocab_file(vocab_file)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def save(self, out_file: str):
        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        white_list: List[str],
        white_list_only: bool,
        pool_size: int,
        chunk_size: int,
        language: str,
    ):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title, count in counter.most_common():
                if title in valid_titles and not title.startswith("Category:"):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                json.dump({"id": ent_id, "entities": [[title, language]], "count": count}, f)
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter
