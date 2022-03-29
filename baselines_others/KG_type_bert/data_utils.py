from __future__ import absolute_import, division, print_function

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


def quality_control_metric(preds, labels, positive_label=1):
    def _auc(preds, labels):
        y = np.array(labels)
        preds = np.array(preds)
        preds = preds[:, positive_label]  # 1 is positive label
        fpr, tpr, thresholds = roc_curve(y, preds, pos_label=1)
        precision, recall, _thresholds = precision_recall_curve(y, preds)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }

    def accuracy(preds, labels):
        outputs = np.argmax(preds, axis=1)
        acc = np.sum(outputs == labels) / len(labels)
        return {"accuracy": acc}

    t1 = _auc(preds, labels)
    t2 = accuracy(preds, labels)
    t1.update(t2)
    return t1


class InputFeatures(object):
    def __init__(self, input_ids, entity_ids, entity_type_ids, input_mask, entity_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.entity_ids = entity_ids
        self.entity_type_ids = entity_type_ids
        self.input_mask = input_mask
        self.entity_mask = entity_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, mention_a=None, mention_b=None,
                 entity_a=None, entity_b=None,
                 entity_a_bias=None, entity_b_bias=None,
                 description_A=None, description_B=None,
                 type_A=None, type_B=None,
                 label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

        self.mention_a = mention_a
        self.mention_b = mention_b

        self.entity_a = entity_a
        self.entity_b = entity_b

        self.entity_a_bias = entity_a_bias
        self.entity_b_bias = entity_b_bias
        self.description_A = description_A
        self.description_B = description_B

        self.type_A = type_A
        self.type_B = type_B

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
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            return json.load(f)


class QualityControlProcessor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type=None):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset_type))), dataset_type)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, dataset_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # note that 1 is positve label
        label_dict = {"0": 0, "1": 1}
        guid = 0
        for (i, line) in enumerate(lines):
            guid = i
            query_and_nn, key_and_nn, label = line
            label = label_dict[label]
            query, query_mention, query_offset, query_entry, query_des, query_type = query_and_nn.split(' [SEP] ')
            keyword, domain, keyword_mention, key_offset, key_entry, key_des, key_type = key_and_nn.split(' [SEP] ')
            keyword = keyword + ' [SEP] ' + domain

            query_offset = int(query_offset) - 1 if int(query_offset) > -1 else -1
            key_offset = int(key_offset) - 1 if int(key_offset) > -1 else -1  # 从零开始编号,-1表示不存在

            keyword = keyword + ' [SEP] ' + query_entry + ' [SEP] ' + key_entry

            examples.append(
                InputExample(guid=guid,
                             text_a=query, text_b=keyword,
                             mention_a=query_mention, mention_b=keyword_mention,
                             entity_a=query_entry, entity_b=key_entry,
                             entity_a_bias=int(query_offset), entity_b_bias=int(key_offset),
                             description_A=query_des, description_B=key_des,
                             type_A=query_type, type_B=key_type,
                             label=label))
        logger.info(f"Finish creating of size {guid + 1}")
        return examples


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_KG-emb_{}_{}_{}_{}'.format(
        dataset_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir,
                                              dataset_type) if evaluate else processor.get_train_examples(args.data_dir,
                                                                                                          dataset_type)
        features = convert_examples_to_features_quality_control(args, examples, label_list, args.max_seq_length,
                                                                tokenizer, output_mode,
                                                                cls_token=tokenizer.cls_token,
                                                                sep_token=tokenizer.sep_token,
                                                                )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_entity_ids = torch.tensor([f.entity_ids for f in features], dtype=torch.long)
    all_entity_type_ids = torch.tensor([f.entity_type_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_entity_mask = torch.tensor([f.entity_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_entity_ids, all_entity_type_ids,
                            all_input_mask, all_entity_mask, all_segment_ids,
                            all_label_ids)
    return dataset


def convert_examples_to_features_quality_control(args, examples, label_list, max_seq_length,
                                                 tokenizer, output_mode,
                                                 cls_token='[CLS]',
                                                 sep_token='[SEP]',
                                                 sequence_a_segment_id=0,
                                                 sequence_b_segment_id=1,
                                                 mask_padding_with_zero=True,
                                                 entity_pad='PAD',
                                                 ):
    label_map = {label: i for i, label in enumerate(label_list)}
    E_ID = load_entityID(args.KG_file, entity_pad=entity_pad)
    E_TYPE = load_entityType(args.TYPE_file, entity_pad=entity_pad)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # if ex_index > 10000:
        #     break
        # 获得mention对应位置的entry id
        if example.entity_a_bias != -1:
            t = example.text_a.split()
            b_1 = example.entity_a_bias
            m_1 = example.mention_a.split()
            a_seg_1, a_seg_2, a_seg_3 = t[:b_1], t[b_1: b_1 + len(m_1)], t[b_1 + len(m_1):]
            a_seg_1 = tokenizer.tokenize(' '.join(a_seg_1))
            a_seg_2 = tokenizer.tokenize(' '.join(a_seg_2))
            a_seg_3 = tokenizer.tokenize(' '.join(a_seg_3))
            tokens_a = a_seg_1 + a_seg_2 + a_seg_3
            entity_tokens_a = [entity_pad] * len(a_seg_1) + [example.entity_a] * len(a_seg_2) + [entity_pad] * len(a_seg_3)
            entity_type_a = [entity_pad] * len(a_seg_1) + [example.type_A] * len(a_seg_2) + [entity_pad] * len(a_seg_3)
        else:
            tokens_a = tokenizer.tokenize(example.text_a)
            entity_tokens_a = [entity_pad] * len(tokens_a)
            entity_type_a = [entity_pad] * len(tokens_a)

        if example.entity_b_bias != -1:
            t = example.text_b.split()
            b_1 = example.entity_b_bias
            m_1 = example.mention_b.split()
            b_seg_1, b_seg_2, b_seg_3 = t[:b_1], t[b_1: b_1 + len(m_1)], t[b_1 + len(m_1):]
            b_seg_1 = tokenizer.tokenize(' '.join(b_seg_1))
            b_seg_2 = tokenizer.tokenize(' '.join(b_seg_2))
            b_seg_3 = tokenizer.tokenize(' '.join(b_seg_3))
            tokens_b = b_seg_1 + b_seg_2 + b_seg_3
            entity_tokens_b = [entity_pad] * len(b_seg_1) + [example.entity_b] * len(b_seg_2) + [entity_pad] * len(b_seg_3)
            entity_type_b = [entity_pad] * len(b_seg_1) + [example.type_B] * len(b_seg_2) + [entity_pad] * len(b_seg_3)
        else:
            tokens_b = tokenizer.tokenize(example.text_b)
            entity_tokens_b = [entity_pad] * len(tokens_b)
            entity_type_b = [entity_pad] * len(tokens_b)

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        _truncate_seq_pair(entity_tokens_a, entity_tokens_b, max_seq_length - 3)
        _truncate_seq_pair(entity_type_a, entity_type_b, max_seq_length - 3)
        tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        entity_tokens = [entity_pad] + entity_tokens_a + [entity_pad] + entity_tokens_b + [entity_pad]
        entity_types = [entity_pad] + entity_type_a + [entity_pad] + entity_type_b + [entity_pad]
        segment_ids = [sequence_a_segment_id] * len([cls_token] + tokens_a) + [sequence_b_segment_id] * len(
            [sep_token] + tokens_b + [sep_token])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # 可能缺失
        entity_ids = [E_ID.get(x, 0) for x in entity_tokens]
        entity_type_ids = [E_TYPE[x] for x in entity_types]
        entity_mask = []
        for x in entity_ids:
            if x == 0:
                entity_mask.append(0)
            else:
                entity_mask.append(1)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        entity_ids += [0] * padding_length
        entity_type_ids += [0] * padding_length
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        entity_mask += ([0] * padding_length)
        segment_ids += ([sequence_a_segment_id] * padding_length)  # 0 0 0 1 1 1 0 0

        # print(len(input_ids), max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(entity_ids) == max_seq_length
        assert len(entity_type_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 20:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("mention tokens: %s" % " ".join([str(x) for x in entity_tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("entity_ids: %s" % " ".join([str(x) for x in entity_ids]))
            logger.info("entity_type_ids: %s" % " ".join([str(x) for x in entity_type_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("entity_mask: %s" % " ".join([str(x) for x in entity_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          entity_ids=entity_ids,
                          entity_type_ids=entity_type_ids,
                          input_mask=input_mask,
                          entity_mask=entity_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


processors = {
    "quality_control": QualityControlProcessor,
}

output_modes = {
    "quality_control": "classification",
}


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    if max_length <= 0:
        tokens_a.clear()
        tokens_b.clear()
    else:
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


def load_entityID(KG_file, entity_pad='pad'):
    E_ID = {entity_pad: 0}
    with open(KG_file, encoding='utf-8') as f:
        id = 1
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 80:
                name = ""
                E_ID[name] = 0
            else:
                name = line[0]
                E_ID[name] = id
                id += 1
    print(id)
    return E_ID


def load_entityType(Type_file, entity_pad='pad'):
    E_TYPE = {entity_pad: 0, "": 0}
    with open(Type_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) == 1:
                continue
            else:
                entityName, type_id = line
                E_TYPE[entityName] = int(type_id)
    return E_TYPE