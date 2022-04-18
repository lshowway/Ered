import csv
import numpy as np
import sys
from io import open
import json
import logging
import os
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def load_and_cache_examples(args, processor, tokenizer, k_tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    input_mode = input_modes[args.task_name]
    output_mode = output_modes[args.task_name]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_use-entity={}_{}_{}_{}_{}_{}'.format(
        args.model_type, args.backbone_model_type, args.knowledge_model_type, args.use_entity,
        args.task_name,
        dataset_type,
        str(args.backbone_seq_length),
        str(args.max_num_entity),
        str(args.knowledge_seq_length)))
    if os.path.exists(cached_features_file):
        logger.warning("===> Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.warning("===> Creating features from dataset file at {}, {}".format(str(args.data_dir), dataset_type))
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, dataset_type)
        else:
            examples = processor.get_train_examples(args.data_dir, dataset_type)
        if input_mode == 'sentence_pair':
            features = convert_examples_to_features_sentence_pair(examples,
                                                                  args.backbone_seq_length, args.knowledge_seq_length, args.max_num_entity,
                                                                  tokenizer, k_tokenizer)
        elif input_mode == 'single_sentence':
            features = convert_examples_to_features_single_sentence(examples, label_list, args.qid_file, args.backbone_seq_length,
                                                   args.knowledge_seq_length, args.max_num_entity, tokenizer, k_tokenizer,)
        elif input_mode == 'entity_sentence':
            features = convert_examples_to_features_entity_typing(args, examples,
                                                                  args.backbone_seq_length, args.knowledge_seq_length,
                                                                  args.max_num_entity, tokenizer, k_tokenizer,
                                                                  )
        elif input_mode == "entity_entity_sentence":
            features = convert_examples_to_features_relation_classification(args, examples,
                                                                            args.backbone_seq_length, args.knowledge_seq_length,
                                                                            args.max_num_entity, tokenizer, k_tokenizer,)
        else:
            features = None
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.warning("===> Saving features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # if isinstance(features, list):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_input_ids_k = torch.tensor([f.k_input_ids for f in features], dtype=torch.long)
    all_k_mask = torch.tensor([f.k_mask for f in features], dtype=torch.long)
    all_input_mask_k = torch.tensor([f.k_input_mask for f in features], dtype=torch.long)
    all_segment_ids_k = torch.tensor([f.k_segment_ids for f in features], dtype=torch.long)
    all_entities = torch.tensor([f.entities for f in features], dtype=torch.long)

    if args.task_name in ['openentity', 'figer']:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if args.task_name in ['fewrel', 'tacred', 'openentity', 'figer']:
        all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,
                                all_input_ids_k, all_k_mask, all_input_mask_k, all_segment_ids_k,
                                all_label_ids, all_entities)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_input_ids_k, all_k_mask, all_input_mask_k, all_segment_ids_k,
                                all_label_ids, all_entities)

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


def convert_examples_to_features_single_sentence(examples, qid_file,
                                                 origin_seq_length, knowledge_seq_length, max_num_entity,
                                                tokenizer, k_tokenizer,
                                                pad_on_left=False,
                                                mask_padding_with_zero=True):

    QID_entityName_dict, QID_description_dict = load_description(qid_file)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        neighbours = [x[0] for x in example.neighbour]

        text_a, label = example.text_a, example.label
        ### ==== append entity ===
        entities = [QID_entityName_dict.get(x, '') for x in neighbours]
        entities = [x for x in entities if x != ""]
        if entities:
            text_a += ' ' + tokenizer.sep_token + ' ' + tokenizer.sep_token.join(entities) + ' ' + tokenizer.sep_token
        ### ==== append entity ===
        tokens = tokenizer.tokenize(text_a)

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]

        segment_ids = [tokenizer.segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = origin_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([tokenizer.pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([tokenizer.pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([tokenizer.pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([tokenizer.pad_token_segment_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length

        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(label_map[example.label])
        # else:
        #     raise KeyError(output_mode)

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label))
        # ==== backbone ====
        # ==== knowledge ====
        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        NULL_description = "The entity does not have description"
        neighbours = [QID_description_dict.get(qid, NULL_description) for qid in neighbours]
        neighbours_mask = [1] * min(max_num_entity, len(neighbours)) + [0] * (max_num_entity - min(max_num_entity, len(neighbours)))
        neighbours = neighbours[: max_num_entity] + ["PAD DESCRIPTION"] * (max_num_entity - len(neighbours))
        for x in neighbours:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1 if mask_padding_with_zero else 0] * len(x_1)  # input_mask
            x_3 = [1 - k_tokenizer.pad_token_type_id] * len(x)
            padding_length = knowledge_seq_length - len(x)
            if pad_on_left:
                x_1 = ([k_tokenizer.pad_token_id] * padding_length) + x_1
                x_2 = ([0 if mask_padding_with_zero else 1] * padding_length) + x_2
                x_3 = ([k_tokenizer.pad_token_type_id] * padding_length) + x_3
            else:
                x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
                x_2 = x_2 + ([0 if mask_padding_with_zero else 1] * padding_length)
                x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            neighbour_one.append(x_1)
            neighbour_att_mask_one.append(x_2)
            neighbour_segment_one.append(x_3)
        # print(len(neighbour_one))  # 如何处理多个entity，多个description
        if ex_index < 10:
            logger.info("*** Example k***")
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


def convert_examples_to_features_relation_classification(args, examples,
                                                         origin_seq_length, knowledge_seq_length, max_num_entity,
                                                        tokenizer, k_tokenizer,
                                                        pad_on_left=False,
                                                        mask_padding_with_zero=True):

    QID_entityName_dict, QID_description_dict = load_description(args.qid_file)
    entity_vocab = get_entity_vocab(args.entity_vocab_file)

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        neighbours = [x[0] for x in example.neighbour]

        entities = [entity_vocab.get(qid, len(entity_vocab)) for qid in neighbours]
        entities = entities[: max_num_entity] + [len(entity_vocab)] * (max_num_entity - len(entities))

        text_a = example.text_a
        subj_start, subj_end, obj_start, obj_end = example.text_b
        # relation = example.label
        if subj_start < obj_start:
            # sub, and then obj (有空格啊，妈的)
            before_sub = text_a[:subj_start].strip()
            tokens = tokenizer.tokenize(before_sub)
            subj_special_start = len(tokens)
            tokens += ['@']
            sub = text_a[subj_start:subj_end + 1].strip()
            tokens += tokenizer.tokenize(sub)
            tokens += ['@']
            between_sub_obj = text_a[subj_end + 1: obj_start].strip()
            tokens += tokenizer.tokenize(between_sub_obj)
            obj_special_start = len(tokens)
            tokens += ['#']
            obj = text_a[obj_start:obj_end + 1].strip()
            tokens += tokenizer.tokenize(obj)
            tokens += ['#']
            after_obj = text_a[obj_end + 1:].strip()
            tokens += tokenizer.tokenize(after_obj)
        else:
            # ojb, and then sub
            before_obj = text_a[:obj_start].strip()
            tokens = tokenizer.tokenize(before_obj)
            obj_special_start = len(tokens)
            tokens += ['#']
            obj = text_a[obj_start: obj_end + 1].strip()
            tokens += tokenizer.tokenize(obj)
            tokens += ['#']
            between_obj_sub = text_a[obj_end + 1: subj_start].strip()
            tokens += tokenizer.tokenize(between_obj_sub)
            subj_special_start = len(tokens)
            tokens += ['@']
            sub = text_a[subj_start:subj_end + 1].strip()
            tokens += tokenizer.tokenize(sub)
            tokens += ['@']
            after_sub = text_a[subj_end + 1:].strip()
            tokens += tokenizer.tokenize(after_sub)

        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]

        subj_special_start += 1 # because of cls_token
        obj_special_start += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
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

        if ex_index < 10:
            logger.info("*** ==> Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(example.label))
        # sure that sub & obj are included in the sequence
        if subj_special_start > origin_seq_length - 1:
            # subj_special_start = origin_seq_length - 10
            subj_special_start = 0
        if obj_special_start > origin_seq_length - 1:
            # obj_special_start = origin_seq_length - 10
            obj_special_start = 0
        # the sub_special_start_id is an array, where the idx of start id is 1, other position is 0.
        subj_special_start_id = np.zeros(origin_seq_length)
        obj_special_start_id = np.zeros(origin_seq_length)
        subj_special_start_id[subj_special_start] = 1
        obj_special_start_id[obj_special_start] = 1
        # ==== backbone ====
        # ==== knowledge ====
        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        NULL_description = "The entity does not have description"
        neighbours = [QID_description_dict.get(qid, NULL_description) for qid in neighbours]
        neighbours_mask = [1] * min(max_num_entity, len(neighbours)) + [0] * (
                    max_num_entity - min(max_num_entity, len(neighbours)))
        neighbours = neighbours[: max_num_entity] + ["PAD DESCRIPTION"] * (max_num_entity - len(neighbours))
        for x in neighbours:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1 if mask_padding_with_zero else 0] * len(x_1)  # input_mask
            x_3 = [0] * len(x)
            padding_length = knowledge_seq_length - len(x)
            if pad_on_left:
                x_1 = ([k_tokenizer.pad_token_id] * padding_length) + x_1
                x_2 = ([0 if mask_padding_with_zero else 1] * padding_length) + x_2
                x_3 = ([k_tokenizer.pad_token_type_id] * padding_length) + x_3
            else:
                x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
                x_2 = x_2 + ([0 if mask_padding_with_zero else 1] * padding_length)
                x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            neighbour_one.append(x_1)
            neighbour_att_mask_one.append(x_2)
            neighbour_segment_one.append(x_3)
        # print(len(neighbour_one))  # 如何处理多个entity，多个description
        if ex_index < 10:
            logger.info("*** Example k***")
            logger.info("tokens_k: %s" % " ".join([str(x_) for x_ in neighbours]))
            logger.info("neighbours_mask: %s" % " ".join([str(x_) for x_ in neighbours_mask]))
            logger.info("input_ids_k: %s" % " ".join([str(x) for x in neighbour_one]))
            logger.info("input_mask_k: %s" % " ".join([str(x) for x in neighbour_att_mask_one]))
            logger.info("segment_ids_k: %s" % " ".join([str(x) for x in neighbour_segment_one]))
            logger.info("entities: %s" % " ".join([str(x) for x in entities]))
        # ==== knowledge ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=example.label,
                          start_id=(subj_special_start_id, obj_special_start_id),
                          k_input_ids=neighbour_one,
                          k_mask=neighbours_mask,  # 表示有几条有效description
                          k_input_mask=neighbour_att_mask_one,
                          k_segment_ids=neighbour_segment_one,  # distilBert没有使用token_type_embedding
                          entities=entities
                          ))

    return features


def convert_examples_to_features_entity_typing(args, examples, origin_seq_length, knowledge_seq_length, max_num_entity,
                                               tokenizer, k_tokenizer,
                                               pad_on_left=False,
                                               mask_padding_with_zero=True):
    QID_entityName_dict, QID_description_dict = load_description(args.qid_file)
    entity_vocab = get_entity_vocab(args.entity_vocab_file)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        neighbours = [x[0] for x in example.neighbour]
        # === 在original input text后面拼上entities
        # if args.use_entity:
            # NULL_entity = "NULL entity"
        entities = [entity_vocab.get(qid, len(entity_vocab)) for qid in neighbours]
        entities = entities[: max_num_entity] + [len(entity_vocab)] * (max_num_entity - len(entities))
            # t = ' ' + tokenizer.sep_token + ' '
            # example.text_a = t.join([example.text_a] + entities)
        # === 在original input text后面拼上entities

        start, end = example.text_b[0], example.text_b[1]
        sentence = example.text_a
        tokens_0_start = tokenizer.tokenize(sentence[:start])
        tokens_start_end = tokenizer.tokenize(sentence[start:end])
        tokens_end_last = tokenizer.tokenize(sentence[end:])
        tokens = [tokenizer.cls_token] + tokens_0_start + tokenizer.tokenize('@') + tokens_start_end + tokenizer.tokenize(
            '@') + tokens_end_last + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]
        start = 1 + len(tokens_0_start)
        if start > origin_seq_length:
            continue
        end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
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


        label_id = example.label

        start_id = np.zeros(origin_seq_length)
        if start >= origin_seq_length:
            start = 0  # 如果entity被截断了，就使用CLS位代替
        start_id[start] = 1
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
            logger.info("start_id: {}".format(start_id))
        # ==== backbone ====
        # ==== knowledge ====
        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        NULL_description = "The entity does not have description"
        neighbours = [QID_description_dict.get(qid, NULL_description) for qid in neighbours]
        neighbours_mask = [1] * min(max_num_entity, len(neighbours)) + [0] * (max_num_entity - min(max_num_entity, len(neighbours)))
        neighbours = neighbours[: max_num_entity] + ["PAD DESCRIPTION"] * (max_num_entity - len(neighbours))
        for x in neighbours:
            x = k_tokenizer.tokenize(x)
            x = [k_tokenizer.cls_token] + x + [k_tokenizer.sep_token]
            x = x[: knowledge_seq_length]
            x_1 = k_tokenizer.convert_tokens_to_ids(x)
            x_2 = [1 if mask_padding_with_zero else 0] * len(x_1)  # input_mask
            x_3 = [0] * len(x)  # segment
            padding_length = knowledge_seq_length - len(x)
            if pad_on_left:
                x_1 = ([k_tokenizer.pad_token_id] * padding_length) + x_1
                x_2 = ([0 if mask_padding_with_zero else 1] * padding_length) + x_2
                x_3 = ([k_tokenizer.pad_token_type_id] * padding_length) + x_3
            else:
                x_1 = x_1 + ([k_tokenizer.pad_token_id] * padding_length)
                x_2 = x_2 + ([0 if mask_padding_with_zero else 1] * padding_length)
                x_3 = x_3 + ([k_tokenizer.pad_token_type_id] * padding_length)
            assert len(x_1) == knowledge_seq_length
            assert len(x_2) == knowledge_seq_length
            assert len(x_3) == knowledge_seq_length
            neighbour_one.append(x_1)
            neighbour_att_mask_one.append(x_2)
            neighbour_segment_one.append(x_3)
        # print(len(neighbour_one))  # 如何处理多个entity，多个description
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens_k: %s" % " ".join([str(x_) for x_ in neighbours]))
            logger.info("neighbours_mask: %s" % " ".join([str(x_) for x_ in neighbours_mask]))
            logger.info("input_ids_k: %s" % " ".join([str(x) for x in neighbour_one]))
            logger.info("input_mask_k: %s" % " ".join([str(x) for x in neighbour_att_mask_one]))
            logger.info("segment_ids_k: %s" % " ".join([str(x) for x in neighbour_segment_one]))
            logger.info("entities: %s" % " ".join([str(x) for x in entities]))
        # ==== knowledge ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          start_id=start_id,
                          k_input_ids=neighbour_one,
                          k_mask=neighbours_mask,  # 表示有几条有效description
                          k_input_mask=neighbour_att_mask_one,
                          k_segment_ids=neighbour_segment_one,  # distilBert没有使用token_type_embedding
                          entities=entities,
                          ))
    return features



class InputFeatures(object):
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,
                 start_id=None,
                 k_input_ids=None, k_mask=None, k_input_mask=None, k_segment_ids=None, label_id=None, entities=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_id = start_id

        self.k_input_ids = k_input_ids
        self.k_mask = k_mask
        self.k_input_mask = k_input_mask
        self.k_segment_ids = k_segment_ids
        self.entities = entities

        self.label_id = label_id



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, neighbour=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.neighbour = neighbour
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
            neighbour = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
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
        # label_set = set()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0] * len(label_list)
            for item in line['labels']:
                # label_set.add(item)
                label[label_list.index(item)] = 1
            neighbour = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
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
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_dev_examples(self, data_dir, dataset_type):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_labels(self):
        relations = dict(zip(TACRED_relations, list(range(len(TACRED_relations)))))

        return relations

    def _create_examples(self, lines, ):
        examples = []
        NA_count = 0
        label_set = self.get_labels()
        no_relation_number = label_set['NA']
        for (i, line) in enumerate(lines):
            guid = i
            # text_a: tokenized words
            text_a = line['text']
            # text_b: other information
            # it is: sub_start, sub_end, obj_start, obj_end
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_b = (line["ents"][0][1], line["ents"][0][2], line["ents"][1][1], line["ents"][1][2])
            label = line['label']
            label = label_set[label]
            neighbour = line['ann']
            if neighbour == "None" or neighbour is None:
                print(line)
            # if label == no_relation_number and dataset_type == 'train':
            #     NA_count += 1
            #     if NA_count < 40000:
            #         examples.append(
            #             InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
            #     else:
            #         continue
            # else:
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
        print('NA_count ', NA_count)
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
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
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
            text_a = line['sent']
            neighbour = [[x[0]] for x in line['ents'] if x[-1] > 0.2]

            label = label_map[line['label']]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, neighbour=neighbour, label=label))
        return examples


def load_description(file):
    # load entity description, e.g., wikidata or wikipedia
    QID_entityName_dict = {}
    QID_description_dict = {}
    with open(file, encoding='utf-8') as f:
        for line in f:
            qid, name, des = line.strip().split('\t')
            QID_entityName_dict[qid] = name
            if des == 'None':
                des = 'NULL'
            QID_description_dict[qid] = des
    return QID_entityName_dict, QID_description_dict

def get_entity_vocab(file):
    entity_list = []
    with open(file, encoding='utf-8') as f:
        for line in f:
            entity = line.strip('\n')
            entity_list.append(entity)
    label_set = dict(zip(entity_list, list(range(len(entity_list)))))  # 1.3m
    return label_set

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
