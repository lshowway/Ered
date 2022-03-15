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



def load_and_cache_examples(args, processor, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_pretrain_{}_{}_{}'.format(
        args.model_type,
        dataset_type,
        str(args.max_seq_length),
        ))
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

        features =  convert_examples_to_features(examples, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ['roberta']),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,)
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.warning("===> Saving features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids)
    return dataset



def convert_examples_to_features(examples,
                                 origin_seq_length,
                                tokenizer,
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

    features = []
    max_num_tokens = origin_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    for (ex_index, x) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        qid, entity_name, description = x.qid, x.entity_name, x.description
        des_mentions_list = x.des_mentions

        # t = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span,
        #      "mention_entity_qid": mention_entity_qid}
        max_end = -1
        for y in des_mentions_list:
            mention = y['mention']
            mention_entity = y['mention_entity']
            mention_qid = y['mention_entity_qid']
            mention_span = y['mention_span']
            if mention_span[1] > max_end:
                max_end = mention_span[1]



        neighbours = [x[0] for x in example.neighbour]

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

        tokens = [cls_token] + tokens + [sep_token]
        tokens = tokens[: origin_seq_length]

        subj_special_start += 1  # because of cls_token
        obj_special_start += 1


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=entity_label
                          ))

    return features



class InputExample(object):
    def __init__(self, guid, qid, entity_name=None, description=None, des_mentions=None):
        self.guid = guid
        self.qid = qid
        self.entity_name = entity_name
        self.description = description
        self.des_mentions = des_mentions



class InputFeatures(object):
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



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



class EntityPredictionProcessor(DataProcessor):
    def __init__(self, data_dir=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.data_dir = data_dir

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines, self.tokenizer)

    def get_dev_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer)

    def get_labels(self):

        return []

    def _create_examples(self, lines, tokenizer, tokenizing_batch_size=32768):
        examples = []
        batch_input, label_list, first_index = [], [], 0
        label_set = {k: v for v, k in enumerate(self.get_labels())}

        for (i, x) in enumerate(lines):
            qid, entity_name, description, des_mentions = \
                x['global_entity_name_qid'], x['global_entity_name'], x['abstract'], x['abstract_mentions']

            # des_mentions = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span,
            #      "mention_entity_qid": mention_entity_qid}

            examples.append(
                InputExample(guid=i, qid=qid, entity_name=entity_name, description=description, des_mentions=des_mentions))


        return examples

