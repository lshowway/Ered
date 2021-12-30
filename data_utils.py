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


def load_and_cache_examples(args, task, tokenizer, k_tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    input_mode = input_modes[task]
    t = 1 if input_mode == 'single_sentence' else 2
    processor = processors[task](tokenizer, k_tokenizer, args.neighbor_num * t)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args.model_type, args.backbone_model_type, args.knowledge_model_type,
        task,
        dataset_type,
        args.use_entity,
        str(args.backbone_seq_length),
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
            features = convert_examples_to_features_sentence_pair(examples, label_list, args.backbone_seq_length,
                                                   args.knowledge_seq_length, tokenizer, k_tokenizer,
                                                   output_mode, input_mode)
        elif input_mode == 'single_sentence':
            features = convert_examples_to_features_single_sentence(examples, label_list, args.backbone_seq_length,
                                                   args.knowledge_seq_length, tokenizer, k_tokenizer,
                                                   output_mode, input_mode)
        elif input_mode == 'entity_sentence':
            features = convert_examples_to_features_entity_typing(examples, args.qid_file, args.backbone_seq_length,
                                                   args.knowledge_seq_length, tokenizer, k_tokenizer,
                                                                  output_mode, input_mode,
                                                                  cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                                  # xlnet has a cls token at the end
                                                                  cls_token=tokenizer.cls_token,
                                                                  cls_token_segment_id=2 if args.model_type in [
                                                                      'xlnet'] else 0,
                                                                  sep_token=tokenizer.sep_token,
                                                                  sep_token_extra=bool(args.model_type in ['roberta']),
                                                                  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                                  pad_on_left=bool(args.model_type in ['xlnet']),
                                                                  # pad on the left for xlnet
                                                                  pad_token=
                                                                  tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                                  pad_token_segment_id=4 if args.model_type in [
                                                                      'xlnet'] else 0,
                                                                  )
        else:
            features = None
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.warning("===> Saving features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if isinstance(features, list):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)

        all_input_ids_k = torch.tensor([f.k_input_ids for f in features], dtype=torch.long)
        all_input_mask_k = torch.tensor([f.k_input_mask for f in features], dtype=torch.long)
        all_segment_ids_k = torch.tensor([f.k_segment_ids for f in features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        all_input_ids = features.input_ids
        all_input_mask = features.input_mask
        all_segment_ids = features.segment_ids
        all_start_ids = features.start_id

        all_input_ids_k = features.k_input_ids
        all_input_mask_k = features.k_input_mask
        all_segment_ids_k = features.k_segment_ids

        all_label_ids = features.label_id

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,
                            all_input_ids_k, all_input_mask_k, all_segment_ids_k,
                            all_label_ids)
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


def convert_examples_to_features_sentence_pair(examples, label_list, origin_seq_length, knowledge_seq_length,
                                 tokenizer, k_tokenizer, output_mode, input_mode, ):
    label_map = {label: i for i, label in enumerate(label_list)}
    input_ids_list, attention_mask_list, segment_ids_list = [], [], []
    neighbour_input_ids_list, neighbour_attention_mask_list, neighbour_segment_ids_list = [], [], []
    labels_list = []
    if input_mode == 'sentence_pair':
        max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=True)  # 32 - 3
    elif input_mode == 'single_sentence':
        max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=False)  # 32 - 2
    else:
        max_num_tokens = -1

    k_max_num_tokens = knowledge_seq_length - k_tokenizer.num_special_tokens_to_add(pair=False)  # 32 - 2
    # 1. tokenize: input & knowledge
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))
        query, key, neighbour, label = example.text_a, example.text_b, example.neighbour, example.label
        if key is not None:
            _truncate_seq_pair(query, key, max_length=max_num_tokens)
        else:
            query = query[: max_num_tokens]
        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=query, token_ids_1=key)
        attention_mask = [1] * len(input_ids)
        if key is not None:
            segment_ids = [tokenizer.pad_token_type_id] * (len(query) + 2) + [1 - tokenizer.pad_token_type_id] * (
                    len(key) + 1)  # roberta 不使用
        else:
            segment_ids = [1 - tokenizer.pad_token_type_id] * len(input_ids)

        input_ids_list.append(input_ids)  # input_ids_one里面有两个list
        attention_mask_list.append(attention_mask)
        segment_ids_list.append(segment_ids)

        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        for x in neighbour:
            x = k_tokenizer.build_inputs_with_special_tokens(token_ids_0=x)
            x = x[: k_max_num_tokens]
            neighbour_one.append(x)
            neighbour_att_mask_one.append([1] * len(x))
            neighbour_segment_one.append([1 - k_tokenizer.pad_token_type_id] * len(x))
        neighbour_input_ids_list.append(neighbour_one)
        neighbour_attention_mask_list.append(neighbour_att_mask_one)
        neighbour_segment_ids_list.append(neighbour_segment_one)
        labels_list.append(label)
    # 2. padding: origina & knowledge
    input_ids_tensor = _tensorize_batch(input_ids_list, tokenizer.pad_token_id, origin_seq_length)  # 这是pad_token_id
    attention_mask_tensor = _tensorize_batch(attention_mask_list, 0, origin_seq_length)
    segment_ids_tensor = _tensorize_batch(segment_ids_list, padding_value=tokenizer.pad_token_type_id,
                                          max_length=origin_seq_length)

    neighbour_input_ids_tensor = _tensorize_batch(neighbour_input_ids_list, padding_value=k_tokenizer.pad_token_id,
                                                  max_length=knowledge_seq_length, neighbour=True)
    neighbour_attention_mask_tensor = _tensorize_batch(neighbour_attention_mask_list, padding_value=0,
                                                       max_length=knowledge_seq_length, neighbour=True)
    neighbour_segment_ids_tensor = _tensorize_batch(neighbour_segment_ids_list,
                                                    padding_value=k_tokenizer.pad_token_type_id,
                                                    max_length=knowledge_seq_length, neighbour=True)
    label_tensor = torch.tensor(labels_list, dtype=torch.long)
    if output_mode == "classification":
        label_id = example.label
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    features = InputFeatures(input_ids=input_ids_tensor,
                             input_mask=attention_mask_tensor,
                             segment_ids=segment_ids_tensor,  # roberta的token_type_num is 1
                             k_input_ids=neighbour_input_ids_tensor,
                             k_input_mask=neighbour_attention_mask_tensor,
                             k_segment_ids=neighbour_segment_ids_tensor,  # distilBert没有使用token_type_embedding
                             label_id=label_tensor,
                             )
    return features


def convert_examples_to_features_single_sentence(examples, label_list, origin_seq_length, knowledge_seq_length,
                                 tokenizer, k_tokenizer, output_mode, input_mode, ):
    label_map = {label: i for i, label in enumerate(label_list)}
    input_ids_list, attention_mask_list, segment_ids_list = [], [], []
    neighbour_input_ids_list, neighbour_attention_mask_list, neighbour_segment_ids_list = [], [], []
    labels_list = []
    if input_mode == 'sentence_pair':
        max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=True)  # 32 - 3
    elif input_mode == 'single_sentence':
        max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=False)  # 32 - 2
    else:
        max_num_tokens = -1

    k_max_num_tokens = knowledge_seq_length - k_tokenizer.num_special_tokens_to_add(pair=False)  # 32 - 2
    # 1. tokenize: input & knowledge
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.warning("Processing example %d of %d" % (ex_index, len(examples)))
        query, key, neighbour, label = example.text_a, example.text_b, example.neighbour, example.label
        if key is not None:
            _truncate_seq_pair(query, key, max_length=max_num_tokens)
        else:
            query = query[: max_num_tokens]
        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=query, token_ids_1=key)
        attention_mask = [1] * len(input_ids)
        if key is not None:
            segment_ids = [tokenizer.pad_token_type_id] * (len(query) + 2) + [1 - tokenizer.pad_token_type_id] * (
                    len(key) + 1)  # roberta 不使用
        else:
            segment_ids = [1 - tokenizer.pad_token_type_id] * len(input_ids)

        input_ids_list.append(input_ids)  # input_ids_one里面有两个list
        attention_mask_list.append(attention_mask)
        segment_ids_list.append(segment_ids)

        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        for x in neighbour:
            x = k_tokenizer.build_inputs_with_special_tokens(token_ids_0=x)
            x = x[: k_max_num_tokens]
            neighbour_one.append(x)
            neighbour_att_mask_one.append([1] * len(x))
            neighbour_segment_one.append([1 - k_tokenizer.pad_token_type_id] * len(x))
        neighbour_input_ids_list.append(neighbour_one)
        neighbour_attention_mask_list.append(neighbour_att_mask_one)
        neighbour_segment_ids_list.append(neighbour_segment_one)
        labels_list.append(label)
    # 2. padding: origina & knowledge
    input_ids_tensor = _tensorize_batch(input_ids_list, tokenizer.pad_token_id, origin_seq_length)  # 这是pad_token_id
    attention_mask_tensor = _tensorize_batch(attention_mask_list, 0, origin_seq_length)
    segment_ids_tensor = _tensorize_batch(segment_ids_list, padding_value=tokenizer.pad_token_type_id,
                                          max_length=origin_seq_length)

    neighbour_input_ids_tensor = _tensorize_batch(neighbour_input_ids_list, padding_value=k_tokenizer.pad_token_id,
                                                  max_length=knowledge_seq_length, neighbour=True)
    neighbour_attention_mask_tensor = _tensorize_batch(neighbour_attention_mask_list, padding_value=0,
                                                       max_length=knowledge_seq_length, neighbour=True)
    neighbour_segment_ids_tensor = _tensorize_batch(neighbour_segment_ids_list,
                                                    padding_value=k_tokenizer.pad_token_type_id,
                                                    max_length=knowledge_seq_length, neighbour=True)
    label_tensor = torch.tensor(labels_list, dtype=torch.long)
    if output_mode == "classification":
        label_id = example.label
    elif output_mode == "regression":
        label_id = float(example.label)
    else:
        raise KeyError(output_mode)

    features = InputFeatures(input_ids=input_ids_tensor,
                             input_mask=attention_mask_tensor,
                             segment_ids=segment_ids_tensor,  # roberta的token_type_num is 1
                             k_input_ids=neighbour_input_ids_tensor,
                             k_input_mask=neighbour_attention_mask_tensor,
                             k_segment_ids=neighbour_segment_ids_tensor,  # distilBert没有使用token_type_embedding
                             label_id=label_tensor,
                             )
    return features


def convert_examples_to_features_entity_typing(examples, qid_file, origin_seq_length, knowledge_seq_length,
                                               tokenizer, k_tokenizer, output_mode, input_mode,
                                               cls_token_at_end=False,
                                               cls_token='[CLS]',
                                               cls_token_segment_id=1,
                                               sep_token='[SEP]',
                                               sep_token_extra=False,
                                               pad_on_left=False,
                                               pad_token=0,
                                               pad_token_segment_id=0,
                                               sequence_a_segment_id=0,
                                               sequence_b_segment_id=1,
                                               mask_padding_with_zero=True):
    # label_map = {label: i for i, label in enumerate(label_list)}
    QID_entityName_dict, QID_description_dict = load_description(qid_file)
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
        tokens = [cls_token] + tokens_0_start + tokenizer.tokenize('@') + tokens_start_end + tokenizer.tokenize(
            '@') + tokens_end_last + [sep_token]
        tokens = tokens[: origin_seq_length]
        start = 1 + len(tokens_0_start)
        if start > origin_seq_length:
            continue
        end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)
        segment_ids = [sequence_a_segment_id] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        padding_length = origin_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length

        if output_mode == "classification":
            label_id = example.label
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        start_id = np.zeros(origin_seq_length)
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
        neighbours = [x[0] for x in example.neighbour]
        neighbour_one, neighbour_att_mask_one, neighbour_segment_one = [], [], []
        neighbours = [QID_description_dict.get(qid, 'NULL') for qid in neighbours]
        neighbours = neighbours[: 1] + (1 - len(neighbours)) * ['NULL']
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
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("tokens_k: %s" % " ".join([str(x_) for x_ in neighbours]))
            logger.info("input_ids_k: %s" % " ".join([str(x) for x in neighbour_one]))
            logger.info("input_mask_k: %s" % " ".join([str(x) for x in neighbour_att_mask_one]))
            logger.info("segment_ids_k: %s" % " ".join([str(x) for x in neighbour_segment_one]))
        # ==== knowledge ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          start_id=start_id,
                          k_input_ids=neighbour_one,
                          k_input_mask=neighbour_att_mask_one,
                          k_segment_ids=neighbour_segment_one,  # distilBert没有使用token_type_embedding
                          ))
    return features


class InputFeatures(object):
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None,
                 start_id=None,
                 k_input_ids=None, k_input_mask=None, k_segment_ids=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_id = start_id

        self.k_input_ids = k_input_ids
        self.k_input_mask = k_input_mask
        self.k_segment_ids = k_segment_ids

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


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self, tokenizer=None, k_tokenizer=None, description_num=None):
        self.tokenizer = tokenizer
        self.k_tokenizer = k_tokenizer
        self.description_num = description_num

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.description_num)

    def get_dev_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.description_num)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, tokenizer, k_tokenizer, neighbor_num, tokenizing_batch_size=32768):
        """Creates examples for the training and dev sets."""
        examples = []
        # note that 1 is positve label
        label_dict = {"0": 0, "1": 1}
        batch_query, batch_neighbour, label_list, first_index = [], [], [], 0
        for (i, line) in enumerate(lines):
            guid = i
            if i == 0:
                continue
            query_and_nn, label = line
            label = label_dict[label]
            label_list.append(label)
            # input text and its corresponding knowledge
            query, _, query_offset, query_entry, query_des = query_and_nn.split(' [SEP] ')

            query_offset = int(query_offset) if int(query_offset) > -1 else -1  # 从零开始编号,-1表示不存在

            # query = query + ' [SEP] ' + query_entry  # 要拼接上entity

            neighbour = [query_des]
            assert len(neighbour) == neighbor_num
            batch_query.append(query)
            batch_neighbour.extend(neighbour)
            if len(batch_neighbour) >= tokenizing_batch_size:
                tokenized_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
                tokenized_neighbour = k_tokenizer.batch_encode_plus(batch_neighbour, add_special_tokens=False)
                t = tokenized_neighbour['input_ids']  # 存在[]
                samples = [[], []]  # query, neighbour
                for j, token_q in enumerate(tokenized_query['input_ids']):
                    samples[0].extend(token_q)  # query
                    start = neighbor_num * j
                    end = neighbor_num * j + neighbor_num
                    samples[-1].extend(t[start: end])  # neighbour (query & key)
                    examples.append(InputExample(guid=first_index, text_a=samples[0], text_b=None,
                                                 neighbour=samples[-1], label=label_list[first_index]))
                    first_index += 1
                    samples = [[], []]
                batch_query, batch_neighbour = [], []
        if len(batch_neighbour) > 0:
            tokenized_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
            tokenized_neighbour = k_tokenizer.batch_encode_plus(batch_neighbour, add_special_tokens=False)
            samples = [[], []]  # query, neighbour
            t = tokenized_neighbour['input_ids']
            print('first index: ', first_index)
            for j, token_q in enumerate(tokenized_query['input_ids']):
                samples[0].extend(token_q)  # query
                start = neighbor_num * j
                end = neighbor_num * j + neighbor_num
                samples[-1].extend(t[start: end])  # neighbour
                if first_index % 200 == 0:
                    print(first_index, label_list[first_index])
                examples.append(InputExample(guid=first_index, text_a=samples[0], text_b=None,
                                             neighbour=samples[-1], label=label_list[first_index]))
                first_index += 1
                samples = [[], []]  # query, neighbour
        logger.info(f"Finish creating of size {first_index}")
        return examples


class QcProcessor(DataProcessor):
    def __init__(self, tokenizer=None, k_tokenizer=None, description_num=None, use_entity=False):
        self.tokenizer = tokenizer
        self.k_tokenizer = k_tokenizer
        self.neighbour_num = description_num
        self.use_entity = use_entity

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_dev_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, tokenizer, k_tokenizer, neighbor_num, tokenizing_batch_size=32768):
        """Creates examples for the training and dev sets."""
        examples = []
        # note that 1 is positve label
        label_dict = {"0": 0, "1": 1}
        batch_query, batch_key, batch_neighbour, label_list, first_index = [], [], [], [], 0
        for (i, line) in enumerate(lines):
            guid = i
            query_and_nn, key_and_nn, label = line
            query, _, _, query_entry, query_des = query_and_nn.split(' [SEP] ')
            keyword, domain, _, _, key_entry, key_des = key_and_nn.split(' [SEP] ')
            keyword = keyword + ' [SEP] ' + domain
            # #####################
            # KFormers中也使用entity信息，第一种直接拼在description前面
            # if self.use_entity:
            #     query_des = query_entry + ' [SEP] ' + query_des
            #     key_des = key_entry + ' [SEP] ' + key_des
            #     neighbour = [query_des, key_des]
            # else:
            #     neighbour = [query_entry, key_entry, query_des, key_des]

            if self.use_entity:
                query = query + ' [SEP] ' + query_entry
                keyword = keyword + ' [SEP] ' + key_entry
            neighbour = [query_des, key_des]

            # #####################

            label = label_dict[label]
            label_list.append(label)
            assert len(neighbour) == neighbor_num
            batch_query.append(query)
            batch_key.append(keyword)
            batch_neighbour.extend(neighbour)
            if len(batch_neighbour) >= tokenizing_batch_size:
                tokenized_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
                tokenized_key = tokenizer.batch_encode_plus(batch_key, add_special_tokens=False)
                tokenized_neighbour = k_tokenizer.batch_encode_plus(batch_neighbour, add_special_tokens=False)
                t = tokenized_neighbour['input_ids']  # 存在[]
                samples = [[], [], []]  # query, key, neighbour
                for j, (token_q, token_k) in enumerate(zip(tokenized_query['input_ids'],
                                                           tokenized_key['input_ids'])):
                    samples[0].extend(token_q)  # query
                    samples[1].extend(token_k)  # key
                    start = neighbor_num * j
                    end = neighbor_num * j + neighbor_num
                    samples[2].extend(t[start: end])  # neighbour (query & key)
                    examples.append(InputExample(guid=first_index, text_a=samples[0], text_b=samples[1],
                                                 neighbour=samples[2], label=label_list[first_index]))
                    first_index += 1
                    samples = [[], [], []]
                batch_query, batch_key, batch_neighbour = [], [], []
        if len(batch_neighbour) > 0:
            tokenized_query = tokenizer.batch_encode_plus(batch_query, add_special_tokens=False)
            tokenized_key = tokenizer.batch_encode_plus(batch_key, add_special_tokens=False)
            tokenized_neighbour = k_tokenizer.batch_encode_plus(batch_neighbour, add_special_tokens=False)
            samples = [[], [], []]  # query, key, neighbour
            t = tokenized_neighbour['input_ids']
            print('first index: ', first_index)
            for j, (token_q, token_k) in enumerate(zip(tokenized_query['input_ids'],
                                                       tokenized_key['input_ids'])):
                samples[0].extend(token_q)  # query
                samples[1].extend(token_k)  # key
                start = neighbor_num * j
                end = neighbor_num * j + neighbor_num
                samples[2].extend(t[start: end])  # neighbour
                if first_index % 200 == 0:
                    print(first_index, label_list[first_index])
                examples.append(InputExample(guid=first_index, text_a=samples[0], text_b=samples[1],
                                             neighbour=samples[2], label=label_list[first_index]))
                first_index += 1
                samples = [[], [], []]  # query, key, neighbour
        logger.info(f"Finish creating of size {first_index}")
        return examples


class OpenentityProcessor(DataProcessor):
    def __init__(self, tokenizer=None, k_tokenizer=None, description_num=None, use_entity=False):
        self.tokenizer = tokenizer
        self.k_tokenizer = k_tokenizer
        self.neighbour_num = description_num
        self.use_entity = use_entity

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, tokenizer, k_tokenizer, neighbor_num, tokenizing_batch_size=32768):
        examples = []
        label_list = ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group']
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0 for item in range(len(label_list))]
            for item in line['labels']:
                label[label_list.index(item)] = 1
            neighbour = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
        return examples


class FigerProcessor(DataProcessor):
    def __init__(self, tokenizer=None, k_tokenizer=None, description_num=None, use_entity=False):
        self.tokenizer = tokenizer
        self.k_tokenizer = k_tokenizer
        self.neighbour_num = description_num
        self.use_entity = use_entity

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer, self.k_tokenizer, self.neighbour_num)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, tokenizer, k_tokenizer, neighbor_num, tokenizing_batch_size=32768):
        examples = []
        label_list = ["/person/artist", "/person", "/transportation", "/location/cemetery", "/language", "/location",
                      "/location/city", "/transportation/road", "/person/actor", "/person/soldier",
                      "/person/politician", "/location/country", "/geography", "/geography/island", "/people",
                      "/people/ethnicity", "/internet", "/internet/website", "/broadcast_network", "/organization",
                      "/organization/company", "/person/athlete", "/organization/sports_team", "/location/county",
                      "/geography/mountain", "/title", "/person/musician", "/event", "/organization/educational_institution",
                      "/person/author", "/military", "/astral_body", "/written_work", "/event/military_conflict", "/person/engineer",
                      "/event/attack", "/organization/sports_league", "/government", "/government/government", "/location/province",
                      "/chemistry", "/music", "/education/educational_degree", "/education", "/building/sports_facility",
                      "/building", "/government_agency", "/broadcast_program", "/living_thing", "/event/election",
                      "/location/body_of_water", "/person/director", "/park", "/event/sports_event", "/law",
                      "/product/ship", "/product", "/product/weapon", "/building/airport", "/software", "/computer/programming_language",
                      "/computer", "/body_part", "/disease", "/art", "/art/film", "/person/monarch", "/game", "/food",
                      "/person/coach", "/government/political_party", "/news_agency", "/rail/railway", "/rail", "/train",
                      "/play", "/god", "/product/airplane", "/event/natural_disaster", "/time", "/person/architect",
                      "/award", "/medicine/medical_treatment", "/medicine/drug", "/medicine", "/organization/fraternity_sorority",
                      "/event/protest", "/product/computer", "/person/religious_leader", "/religion", "/religion/religion",
                      "/building/theater", "/biology", "/livingthing", "/livingthing/animal", "/finance/currency", "/finance",
                      "/organization/airline", "/product/instrument", "/location/bridge", "/building/restaurant", "/medicine/symptom",
                      "/product/car", "/person/doctor", "/metropolitan_transit", "/metropolitan_transit/transit_line", "/transit",
                      "/product/spacecraft", "/broadcast", "/broadcast/tv_channel", "/building/library", "/education/department", "/building/hospital"]
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0] * len(label_list)
            for item in line['labels']:
                label[label_list.index(item)] = 1
            neighbour = line['ents']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, neighbour=neighbour, label=label))
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

processors = {
    "sst2": Sst2Processor,
    "quality_control": QcProcessor,
    "open_entity": OpenentityProcessor,
    "figer": FigerProcessor,
}

output_modes = {
    "qqp": "classification",
    "qnli": "classification",
    "wnli": "classification",
    "sst2": "classification",
    "quality_control": "classification",
    "open_entity": "classification",
    "figer": "classification",
}

input_modes = {
    "qqp": "sentence_pair",
    "qnli": "sentence_pair",
    "wnli": "sentence_pair",
    "sst2": "single_sentence",
    "quality_control": "sentence_pair",
    "open_entity": "entity_sentence",
    "figer": "entity_sentence",
}
