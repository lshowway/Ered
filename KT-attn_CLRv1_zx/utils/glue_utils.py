# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import collections
import numpy as np
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, precision_recall_curve, auc, roc_curve

logger = logging.getLogger(__name__)


def pred_argmax(out):
    return np.argmax(out, axis=1).reshape(-1)


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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExampleMultiTask(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, task_id=None):
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
        self.label = label
        self.task_id = task_id


class InputFeaturesMultiTask(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, task_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.task_ids = task_ids


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

    def get_test_examples(self, data_dir):
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
            logger.info("Read %d lines from %s" % (len(lines), input_file))
            return lines

    def get_pred(self, out):
        # default: classification
        lbl_list = self.get_labels()
        return [lbl_list[p] for p in pred_argmax(out).tolist()]


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, predict_file):
        """See base class."""
        return self._create_examples(self._read_tsv(predict_file), "predict")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, data_type=None):
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

            # entity 拼接在keyword后面
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


class AdsProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "CLRv1_format.train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["Bad", "Good"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class AdsMultiTaskProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "CLRv1_format.train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["Bad", "Good"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            task = line[3]
            examples.append(InputExampleMultiTask(guid=guid, text_a=text_a, text_b=text_b, label=label, task_id=task))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test_matched", is_test=True)

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type, is_test=False):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            if is_test:
                label = self.get_labels()[0]
            else:
                label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test_mismatched", is_test=True)


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "CLRv1_format.train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(args, examples, origin_seq_length, max_seq_length,
                                 tokenizer, cls_token, sep_token, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    length_counter = collections.defaultdict(int)
    a_idx_set, b_idx_set = set(), set()

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 20000 == 0:
            # 因为只有local_rank==0才load data，所以这里只打印一次
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))
        if args.clip_data:
            if ex_index > 1000:
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
        if entity_a_bias >= len(tokens_a):
            entity_a_bias = -1
        if entity_b_bias >= len(tokens_b):
            entity_b_bias = -1
        # token a
        tokens = [cls_token] + tokens_a + [sep_token]
        seg_id = 0
        segment_ids = [seg_id] * (len(tokens_a) + 2)
        seg_id = (seg_id + 1) % 2
        #  tokens_b:
        tokens += tokens_b + [sep_token]
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
                tokens += description_A + [sep_token]
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
                if len(description_B) > 0:
                    tokens += description_B + [sep_token]
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

        mask_index = [t0, t1, t2, a_idx, b_idx]
        # Zero-pad up to the sequence length.
        segment_ids += [seg_id] * (max_seq_length - len(input_ids))
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding

        if ex_index < 10:
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
        b_idx_set.add(b_idx)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=mask_index,
                          segment_ids=segment_ids,
                          label_id=label))
    print(a_idx_set, b_idx_set)
    return features



def convert_examples_to_features_multitask(examples, label_list, max_seq_length, tokenizer, cls_token, sep_token,
                                           output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    # label_map = {label: i for i, label in enumerate(label_list)}

    length_counter = collections.defaultdict(int)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            length_counter[len(tokens_a) + len(tokens_b)] += 1
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            length_counter[len(tokens_a)] += 1
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = [cls_token] + tokens_a + [sep_token]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label = int(example.label)
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        task_id = int(example.task_id)

        if ex_index < 20:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % (example.label))
            logger.info("task_id: %s" % (example.task_id))

        features.append(
            InputFeaturesMultiTask(input_ids=input_ids,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids,
                                   label_id=label,
                                   task_ids=task_id))

    max_len = max(length_counter.keys())
    a = 0
    while a < max_len:
        cc = 0
        for i in range(10):
            cc += length_counter[a + i]

        logger.info("%d ~ %d = %d" % (a, a + 10, cc))
        a += 10

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


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def auc_metrics(preds, labels):
    y = np.array(labels)
    pred = np.array(preds)
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    precision, recall, _thresholds = precision_recall_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return matthews_corrcoef(labels, preds)
    elif task_name == "sst-2":
        return simple_accuracy(preds, labels)
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return simple_accuracy(preds, labels)
    elif task_name == "mnli-mm":
        return simple_accuracy(preds, labels)
    elif task_name == "qnli":
        return simple_accuracy(preds, labels)
    elif task_name == "rte":
        return simple_accuracy(preds, labels)
    elif task_name == "wnli":
        return simple_accuracy(preds, labels)
    # elif task_name == "ads":
    #   return auc_metrics(preds, labels)
    elif task_name == 'ads':
        t1 = auc_metrics(preds, labels)
        t2 = simple_accuracy(preds, labels)
        t1['accuracy'] = t2
        return t1
    elif task_name == "ads-multitask":
        return auc_metrics(preds, labels)
    else:
        raise KeyError(task_name)


auc_metrics_tasks = ["ads-multitask", "ads"]

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "ads": MrpcProcessor,
    "ads-multitask": AdsMultiTaskProcessor
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "ads": "classification",
    "ads-multitask": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "ads": 2,
    "ads-multitask": 2
}
