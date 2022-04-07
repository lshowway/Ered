import csv
import numpy as np
import sys
from io import open
import json
import logging
import os
import torch
from more_itertools import locate
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc, roc_curve, accuracy_score
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)



def load_and_cache_examples(args, processor, tokenizer, dataset_type, evaluate=False):
    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_wikipedia_{}_{}_{}'.format(
        args.model_type,
        dataset_type,
        str(args.max_seq_length),
        ))
    if os.path.exists(cached_features_file):
        logger.warning("===> Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.warning("===> Creating features from dataset file at {}, {}".format(str(args.data_dir), dataset_type))
        if evaluate:
            features = processor.get_dev_examples(args.data_dir+'/BB', dataset_type)
        else:
            features = processor.get_train_examples(args.data_dir+'/BB', dataset_type)

        # features =  convert_examples_to_features(examples, args.max_seq_length, tokenizer)
        if args.local_rank in [-1, 0]:
            logger.warning("===> Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            logger.warning("===> Saved features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)

    mention_span = torch.tensor([f.mention_span for f in features], dtype=torch.float)
    mention_entity = torch.tensor([f.mention_entity for f in features], dtype=torch.long)
    description_entity = torch.tensor([f.description_entity for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids, mlm_labels,
                            mention_span, mention_entity, description_entity)

    return dataset



# class InputExample(object):
#     def __init__(self, guid, qid, entity_name=None, description=None, des_mentions=None):
#         self.guid = guid
#         self.qid = qid
#         self.entity_name = entity_name
#         self.description = description
#         self.des_mentions = des_mentions



class InputFeatures(object):
    def __init__(self, input_ids=None, mlm_labels=None,
                 mention_span=None, mention_entity=None, description_entity=None,):
        self.input_ids = input_ids
        # self.input_mask = input_mask
        # self.segment_ids = segment_ids
        self.mlm_labels = mlm_labels

        self.mention_span = mention_span
        self.mention_entity = mention_entity
        self.description_entity = description_entity

        # self.entity_labels = entity_labels



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, ):
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
            data = [json.loads(line) for line in f]
            return data



class EntityPredictionProcessor(DataProcessor):
    def __init__(self, data_dir=None, tokenizer=None, max_seq_length=None):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length

    def get_train_examples(self, data_dir, dataset_type=None):

        files = os.listdir(data_dir)
        all_lines = []
        t = self.get_labels()
        label_set = dict(zip(t, list(range(len(t))))) # 1.3m

        for file in files:
            print(file)
            lines = self._read_json(os.path.join(data_dir, file))
            all_lines.extend(lines)
        examples =  self._create_examples(all_lines, label_set=label_set)

        print('[total] number of pre-training samples: ', len(examples))

        return examples


    def get_labels(self):
        if os.path.exists(os.path.join(self.data_dir, 'entity_vocab.tsv')):
            entity_list = []
            with open(os.path.join(self.data_dir, 'entity_vocab.tsv'), encoding='utf-8') as f:
                for line in f:
                    entity = line.strip('\n')
                    entity_list.append(entity)
        else:
            entity_count_dict = defaultdict(int)
            files = os.listdir(self.data_dir+'/BB')
            # all_lines = []
            for file in files:
                print(file)
                lines = self._read_json(os.path.join(self.data_dir+'/BB', file))
                # all_lines.extend(lines)
                for x in lines:
                    title, annotation = x['title'], x['annotation']
                    entity_count_dict[title] += 1
                    for anchor, entity in annotation.items():
                        entity_count_dict[entity] += 1

            entity_list = []
            for entity, entity_count in entity_count_dict.items():
                if 10 <= entity_count <= 5000 and len(entity.split('_')) < 6:
                    entity_list.append(entity)

            with open(os.path.join(self.data_dir, 'entity_vocab.tsv'), 'w', encoding='utf-8') as fw:
                logger.info("Writing entity vocabulary into: %s".format(self.data_dir + 'entity_vocab.tsv'))
                for e in entity_list:
                    fw.write(e + '\n')

        print('The total number of entity vocabulary is: ', len(entity_list))
        logger.info('The total number of entity vocabulary is: {}'.format(len(entity_list)))
        return entity_list

    def _create_examples(self, lines, label_set):
        features = []
        remove_count = 0
        filtered_anchor = 0
        for (i, x) in enumerate(lines):
            if i % 100000 == 0:
                print('[total wikipedia papges] Processing  features: ', i)

            # if i > 1000:
            #     break

            title, text, annotation = x['title'], x['text'], x['annotation']
            if title in label_set:
                title_id = label_set[title]  # 不可以报错
            else:
                continue  # only about 1.3 million entities are considered
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            start_indexes = list(locate(tokens, lambda x: x == '<s>'))
            end_indexes = list(locate(tokens, lambda x: x == '</s>'))  # 这儿确实有bug，因为document中很大概率有@符号，改成<s></s>符号
            if len(start_indexes) != len(end_indexes): # 这些就直接扔掉了
                remove_count += 1
                continue
            for start, end in zip(start_indexes, end_indexes):
                anchor_len = end - start
                l_len = (self.max_seq_length - anchor_len - 1) // 2 # 包括@

                input_ids = [self.tokenizer.cls_token_id] + token_ids[start - l_len: start]
                inputs = [self.tokenizer.cls_token] + tokens[start - l_len: start]

                span_id = np.zeros(self.max_seq_length)
                new_start = len(inputs)
                span_id[new_start] = 1

                input_ids += token_ids[start: end]
                inputs += tokens[start: end]

                new_end = len(inputs)
                if new_end >= self.max_seq_length:
                    filtered_anchor += 1
                    continue
                span_id[new_end] = 1

                input_ids += token_ids[end: self.max_seq_length - 1] # 减去cls
                inputs += tokens[end: self.max_seq_length - 1]
                # pad,还加attn_mask嘛
                # if len(input_ids) < self.max_seq_length:
                #     is_mask += 1
                input_ids = input_ids + ([self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids)))

                mlm_labels = [-1] * self.max_seq_length
                mlm_labels[new_start+1: new_end] = token_ids[start+1: end]

                anchor = self.tokenizer.convert_tokens_to_string(tokens[start+1: end])
                # print(start, end, new_start, new_end, inputs)
                # print(anchor)
                # print(annotation)
                if anchor in annotation:
                    entity = annotation[anchor]
                    if entity in label_set:
                        anchor_label = label_set[entity] # linked entity
                    else:
                        continue
                else:
                    filtered_anchor += 1
                    continue

                # print(len(input_ids), len(mlm_labels), len(span_id))
                assert len(input_ids) == self.max_seq_length
                assert len(mlm_labels) == self.max_seq_length
                assert len(span_id) == self.max_seq_length

                features.append(InputFeatures(input_ids=input_ids,
                              mlm_labels=mlm_labels,
                              mention_span=span_id,
                              mention_entity=anchor_label,  # 这需要转换成idx表示的
                              description_entity=title_id,  # # 这需要转换成idx表示的
                              ))
        # print('removed counts: ', remove_count)
        # print('filtered anchor: ', filtered_anchor)
        print('the total number of pre-training samples of this file: ', len(features))
        return features