from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import sys
from io import open
import json
import logging
import os
import collections
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
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None,
                 entity_a=None, entity_b=None,
                 entity_a_bias=None, entity_b_bias=None,
                 description_A=None, description_B=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
      sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
      Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
      specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

        self.entity_a = entity_a
        self.entity_b = entity_b

        self.entity_a_bias = entity_a_bias
        self.entity_b_bias = entity_b_bias
        self.description_A = description_A
        self.description_B = description_B
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
            query, _, query_offset, query_entry, query_des = query_and_nn.split(' [SEP] ')
            keyword, domain, _, key_offset, key_entry, key_des = key_and_nn.split(' [SEP] ')
            keyword = keyword + ' [SEP] ' + domain

            query_offset = int(query_offset) - 1 if int(query_offset) > -1 else -1
            key_offset = int(key_offset) - 1 if int(key_offset) > -1 else -1  # 从零开始编号,-1表示不存在

            # query = query + ' [SEP] ' + query_entry
            # keyword = keyword + ' [SEP] ' + key_entry

            keyword = keyword + ' [SEP] ' + query_entry + ' [SEP] ' + key_entry

            examples.append(
                InputExample(guid=guid,
                             text_a=query, text_b=keyword,
                             entity_a=query_entry, entity_b=key_entry,
                             entity_a_bias=int(query_offset), entity_b_bias=int(key_offset),
                             description_A=query_des, description_B=key_des,
                             label=label))
        logger.info(f"Finish creating of size {guid + 1}")
        return examples


class Sst2Processor(DataProcessor):
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
            if i == 0:
                continue
            guid = i
            query_and_nn, label = line
            label = label_dict[label]
            query, _, query_offset, query_entry, query_des = query_and_nn.split(' [SEP] ')

            query_offset = int(query_offset) if int(query_offset) > -1 else -1  # 从零开始编号,-1表示不存在

            query = query + ' [SEP] ' + query_entry

            examples.append(
                InputExample(guid=guid,
                             text_a=query, text_b=None,
                             entity_a=query_entry, entity_b=None,
                             entity_a_bias=int(query_offset), entity_b_bias=None,
                             description_A=query_des, description_B=None,
                             label=label))
        logger.info(f"Finish creating of size {guid + 1}")
        return examples


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    input_mode = input_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir,
                                        'cached_KT-attn-bert-_{}_{}_origin_seq_length={}_max_seq_length={}'.format(
                                            task,
                                            dataset_type,
                                            str(args.origin_seq_length),
                                            str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(args.data_dir, dataset_type) if evaluate \
            else processor.get_train_examples(args.data_dir, dataset_type)
        if input_mode == 'sentence_pair':
            features = convert_examples_to_features_sentence_pair(args, examples, args.origin_seq_length,
                                                                  args.max_seq_length, tokenizer,
                                                                  output_mode=output_mode, )
        else:
            features = convert_examples_to_features_sentence_single(args, examples, args.origin_seq_length,
                                                                    args.max_seq_length, tokenizer,
                                                                    output_mode=output_mode, )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def convert_examples_to_features_sentence_pair(args, examples, origin_seq_length, max_seq_length, tokenizer,
                                               output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    length_counter = collections.defaultdict(int)
    a_idx_set, b_idx_set = set(), set()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 20000 == 0:
            # 因为只有local_rank==0才load data，所以这里只打印一次
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))
        if args.clip_data:
            if ex_index > 10000:
                break

        # sentence A
        tokens_a = tokenizer.tokenize(example.text_a)
        entity_a_bias = example.entity_a_bias  # sentence A

        text_a = example.text_a.split()
        if entity_a_bias not in [-1, 0]:  # entry如果不存在或者在首位，暂不处理
            text_a_before_en = tokenizer.tokenize(' '.join(text_a[:entity_a_bias]))  # 要+1，因为最后一个不包括
            entity_a_bias = len(text_a_before_en)  # if len(text_a_before_en) > 0 else -1
        if example.description_A:
            description_A = tokenizer.tokenize(example.description_A)

        # sentence B
        tokens_b = tokenizer.tokenize(example.text_b)
        entity_b_bias = example.entity_b_bias  # sentence B

        text_b = example.text_b.split()
        if entity_b_bias not in [-1, 0]:
            text_b_before_en = tokenizer.tokenize(' '.join(text_b[:entity_b_bias]))
            entity_b_bias = len(text_b_before_en) if len(text_b_before_en) > 0 else -1
        if example.description_B:
            description_B = tokenizer.tokenize(example.description_B)
        # 截断操作 (可能把entity截断（全截断，或者截断一半）)
        length_counter[len(tokens_a) + len(tokens_b) + len(description_A) + len(description_B)] += 1
        _truncate_seq_pair(tokens_a, tokens_b, origin_seq_length - 3)  # 如果qk占满了就不用des
        if len(tokens_a) + len(tokens_b) == max_seq_length - 3:
            description_B = []
            description_A = []
        else:
            if example.description_A and example.description_B:
                _truncate_seq_pair(description_A, description_B, max_seq_length - len(tokens_a) - len(tokens_b) - 3 - 2)
            elif example.description_A:
                description_A = description_A[: max_seq_length - len(tokens_a) - len(tokens_b) - 3 - 1]
                description_B = []
            elif example.description_B:
                description_A = []
                description_B = description_B[: max_seq_length - len(tokens_a) - len(tokens_b) - 3 - 1]
            else:
                description_A = []
                description_B = []
        # print(len(description_B),  len(description_A), max_seq_length - len(tokens_a) - len(tokens_b) - 3 - 2)
        # assert len(description_B) + len(description_A) <= max_seq_length - len(tokens_a) - len(tokens_b) - 3 - 2

        if entity_a_bias >= len(tokens_a):
            entity_a_bias = -1
        if entity_b_bias >= len(tokens_b):
            entity_b_bias = -1
        # token a
        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        seg_id = 0
        segment_ids = [seg_id] * (len(tokens_a) + 2)
        seg_id = (seg_id + 1) % 2
        #  tokens_b:
        tokens += tokens_b + [tokenizer.sep_token]
        segment_ids += [seg_id] * (len(tokens_b) + 1)
        seg_id = (seg_id + 1) % 2

        t0 = len(tokens_a) + len(tokens_b) + 3  # qk的index
        if example.entity_a_bias == -1:  # 连entity都没有
            a_idx = -1
            t1 = -1
        else:
            a_idx = entity_a_bias + 1  # 1 is for +CLS

            # description A
            if len(description_A) > 0:
                tokens += description_A + [tokenizer.sep_token]
                segment_ids += [seg_id] * (len(description_A) + 1)
                seg_id = (seg_id + 1) % 2

                # entity A: inner description and A to its description
                t1 = t0 + len(description_A) + 1  # entity description+sep的长度
            else:
                t1 = t0

        if example.entity_b_bias == -1:  # 连entity都没有
            b_idx = -1
            t2 = -1
        else:
            b_idx = len(tokens_a) + 2 + entity_b_bias
            if t1 == max_seq_length:
                t2 = -1
            else:
                # if description B:
                if len(description_B) > 0:
                    tokens += description_B + [tokenizer.sep_token]
                    segment_ids += [seg_id] * (len(description_B) + 1)
                    seg_id = (seg_id + 1) % 2

                    # entity B: inner description and B to its description
                    if t1 == -1:
                        t2 = t0 + len(description_B) + 1
                    else:
                        t2 = t1 + len(description_B) + 1
                else:
                    if t1 != -1 and t1 != max_seq_length:
                        t2 = t1
                    else:
                        t2 = t0

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # input_ids = input_ids[: max]

        mask_index = [t0, t1, t2, a_idx, b_idx]
        # Zero-pad up to the sequence length.
        segment_ids += [seg_id] * (max_seq_length - len(input_ids))
        # segment_ids = [0] * (len(tokens_a)+3+len(tokens_b)) + [1] * (max_seq_length - len(tokens_a) -len(tokens_b) - 3)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        if ex_index < 0:
            # if b_idx < 0 or a_idx < 0:
            # if a_idx >= max_seq_length or b_idx >= max_seq_length:
            # if t1 < 1 or t1 < t0:
            # if t2 < 1 or t2 < t0 or t2 < t1:
            # if example.entity_a_bias == -1 and (a_idx != -1 or t1 != -1):
            # if example.entity_b_bias == -1 and (b_idx != -1 or t2 != -1):
            # if t2 != -1 and t2 <= t0:
            # if t1 != -1 and t1 <= t0:
            # if t1 != -1 and t2 != -1 and t2 <= t1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))

            logger.info("text a: %s, %s, %s, %s" % (
            example.text_a, example.entity_a, example.entity_a_bias, example.description_A))

            logger.info("text b: %s, %s, %s, %s" % (
            example.text_b, example.entity_b, example.entity_b_bias, example.description_B))
            logger.info("input_mask: %s" % " ".join([str(x) for x in mask_index]))

        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(mask_index) == 5  # 五个关键点索引
        assert t0 <= max_seq_length
        assert t0 <= t1 <= max_seq_length or t1 == -1
        # print(mask_index)
        assert (t1 <= t2 <= max_seq_length and t2 >= t0) or t2 == -1
        assert a_idx <= max_seq_length - 1
        assert b_idx <= max_seq_length - 1
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label = example.label
        # elif output_mode == "regression":
        #   label = float(example.label)
        # else:
        #   raise KeyError(output_mode)

        a_idx_set.add(a_idx)
        b_idx_set.add(b_idx)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=mask_index,
                          segment_ids=segment_ids,
                          label_id=label))
    print(a_idx_set, b_idx_set)
    return features


def convert_examples_to_features_sentence_single(args, examples, origin_seq_length, max_seq_length, tokenizer,
                                                 output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    # length_counter = collections.defaultdict(int)
    a_idx_set = set()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 20000 == 0:
            # 因为只有local_rank==0才load data，所以这里只打印一次
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))
        if args.clip_data:
            if ex_index > 10000:
                break

        # sentence A
        tokens_a = tokenizer.tokenize(example.text_a)
        entity_a_bias = example.entity_a_bias  # sentence A

        text_a = example.text_a.split()
        if entity_a_bias not in [-1, 0]:  # entry如果不存在或者在首位，暂不处理
            text_a_before_en = tokenizer.tokenize(' '.join(text_a[:entity_a_bias]))  # 要+1，因为最后一个不包括
            entity_a_bias = len(text_a_before_en)  # if len(text_a_before_en) > 0 else -1
        if example.description_A:
            description_A = tokenizer.tokenize(example.description_A)

        # 截断操作 (可能把entity截断（全截断，或者截断一半）)
        tokens_a = tokens_a[: origin_seq_length - 2]  # 如果qk占满了就不用des
        if len(tokens_a) == max_seq_length - 2:
            description_A = []
        else:
            if example.description_A:
                description_A = description_A[: max_seq_length - len(tokens_a) - 2 - 1]
            else:
                description_A = []

        if entity_a_bias >= len(tokens_a):
            entity_a_bias = -1
        # token a
        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        seg_id = 0
        segment_ids = [seg_id] * (len(tokens_a) + 2)
        seg_id = (seg_id + 1) % 2

        t0 = len(tokens_a) + 2  # qk的index
        if example.entity_a_bias == -1:  # 连entity都没有
            a_idx = -1
            t1 = -1
        else:
            a_idx = entity_a_bias + 1  # 1 is for +CLS

            # description A
            if len(description_A) > 0:
                tokens += description_A + [tokenizer.sep_token]
                segment_ids += [seg_id] * (len(description_A) + 1)
                seg_id = (seg_id + 1) % 2

                # entity A: inner description and A to its description
                t1 = t0 + len(description_A) + 1  # entity description+sep的长度
            else:
                t1 = t0

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        t2, b_idx = -1, -1
        mask_index = [t0, t1, t2, a_idx, b_idx]
        # Zero-pad up to the sequence length.
        segment_ids += [seg_id] * (max_seq_length - len(input_ids))
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        if ex_index < 20:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))

            logger.info("text a: %s, %s, %s, %s" % (
            example.text_a, example.entity_a, example.entity_a_bias, example.description_A))

            logger.info("text b: %s, %s, %s, %s" % (
            example.text_b, example.entity_b, example.entity_b_bias, example.description_B))
            logger.info("input_mask: %s" % " ".join([str(x) for x in mask_index]))

        assert len(input_ids) == max_seq_length
        assert len(mask_index) == 5  # 五个关键点索引
        assert t0 <= max_seq_length
        assert t0 <= t1 <= max_seq_length or t1 == -1
        assert (t1 <= t2 <= max_seq_length and t2 >= t0) or t2 == -1
        assert a_idx <= max_seq_length - 1
        assert b_idx <= max_seq_length - 1
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label = example.label
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        a_idx_set.add(a_idx)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=mask_index,
                          segment_ids=segment_ids,
                          label_id=label))
    print(a_idx_set)
    return features


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


processors = {
    "quality_control": QualityControlProcessor,
    "sst2": Sst2Processor,
}

output_modes = {
    "quality_control": "classification",
    "sst2": "classification",
}

input_modes = {
    "quality_control": "sentence_pair",
    "sst2": "sentence_single",
}
