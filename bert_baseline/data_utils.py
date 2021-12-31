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
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # note that 1 is positve label
        label_list = {"0": 0, "1": 1}
        for (i, line) in enumerate(lines):
            guid = i
            # print(line)
            text_a, text_b, l = line
            label = label_list[l]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
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

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # note that 1 is positve label
        label_list = {"0": 0, "1": 1}
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            # print(line)
            text_a, l = line
            label = label_list[l]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    input_mode = input_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
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
        if input_mode == 'sentence_pair':
            features = convert_examples_to_features_pair(examples, label_list, args.max_seq_length, tokenizer,
                                                         output_mode,
                                                         cls_token=tokenizer.cls_token,
                                                         sep_token=tokenizer.sep_token,
                                                         pad_token=
                                                         tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                         pad_token_segment_id=0,
                                                         )
        else:
            features = convert_examples_to_features_single(examples, label_list, args.max_seq_length, tokenizer,
                                                           output_mode,
                                                           cls_token=tokenizer.cls_token,
                                                           sep_token=tokenizer.sep_token,
                                                           pad_token=
                                                           tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                           pad_token_segment_id=1,
                                                           )
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


def convert_examples_to_features_pair(examples, label_list, max_seq_length,
                                      tokenizer, output_mode,
                                      cls_token='[CLS]',
                                      sep_token='[SEP]',
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      sequence_a_segment_id=0,
                                      sequence_b_segment_id=1,
                                      mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = [cls_token] + tokens_a + [sep_token] + tokens_b + [sep_token]
        segment_ids = [sequence_a_segment_id] * len([cls_token] + tokens_a) + [sequence_b_segment_id] * len(
            [sep_token] + tokens_b + [sep_token])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([sequence_a_segment_id] * padding_length)  # 0 0 0 1 1 1 0 0

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_examples_to_features_single(examples, label_list, max_seq_length,
                                        tokenizer, output_mode,
                                        cls_token='[CLS]',
                                        sep_token='[SEP]',
                                        pad_token=0,
                                        pad_token_segment_id=1,
                                        sequence_a_segment_id=0,
                                        sequence_b_segment_id=None,
                                        mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_a = tokens_a[: max_seq_length - 2]

        tokens = [cls_token] + tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len([cls_token] + tokens_a + [sep_token])
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([pad_token] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)  # 0 0 0 1 1 1 0 0

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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
