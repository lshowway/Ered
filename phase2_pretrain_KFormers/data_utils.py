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
from datetime import timedelta

logger = logging.getLogger(__name__)



def load_and_cache_examples(args, processor, dataset_type, evaluate=False, start=0, end=-1):

    # dataset_type: train, dev, test
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    features = processor.get_train_examples(args.data_dir, args.local_rank, dataset_type, start=start, end=end)
    if args.local_rank == 0:
        torch.distributed.barrier()
    # if args.local_rank == 0 and not evaluate:
    #     torch.distributed.monitored_barrier(timeout=timedelta(hours=8))  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # print('99, {}'.format(args.local_rank)) # 0 1 2 3
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



class EntityPredictionProcessor():
    def __init__(self, data_dir=None, tokenizer=None, max_seq_length=None):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length

    def _read_json(cls, input_file):
        with open(input_file, 'r', encoding='utf8') as f:
            data = [json.loads(line) for line in f]
            return data

    def get_train_examples(self, data_dir, local_rank, dataset_type=None, start=0, end=-1):

        read_dir = data_dir + '/DD_64'
        all_features = []

        if not os.path.exists(read_dir):
            os.makedirs(read_dir)

            json_dir = data_dir + '/BB'
            json_files = os.listdir(json_dir)
            t = self.get_labels()
            label_set = dict(zip(t, list(range(len(t)))))  # 1.3m

            for file in json_files:
                print(file)
                if os.path.exists(os.path.join(read_dir, 'cached_%s'%file)):
                    features = torch.load(read_dir, 'cached_%s'%file)
                else:
                    lines = self._read_json(os.path.join(json_dir, file))
                    features = self._create_examples(lines, label_set=label_set)
                    if local_rank in [-1, 0]:
                        torch.save(features, os.path.join(read_dir, 'cached_%s'%file))

                all_features.extend(features)
        else:
            # load cached
            files = os.listdir(read_dir)

            for i, file in enumerate(files[start: end]):
                logging.info('processing {} {} {}'.format(i, local_rank, file))
                features = torch.load(os.path.join(read_dir, file), map_location='cpu')
                all_features.extend(features)


        logging.info('[total] number of pre-training samples: {}'.format(len(all_features)))

        return all_features


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
                logging.info(file)
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

        # print('The total number of entity vocabulary is: ', len(entity_list))
        logger.info('The total number of entity vocabulary is: {}'.format(len(entity_list)))
        return entity_list

    def _create_examples(self, lines, label_set):
        features = []
        remove_count = 0
        filtered_anchor = 0
        for (i, x) in enumerate(lines):
            if i % 10000 == 0:
                logging.info('Processing  features: {}'.format(i))

            # if i > 100:
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

                input_ids += token_ids[end: end + self.max_seq_length - len(input_ids)] # 减去cls
                inputs += tokens[end: end + self.max_seq_length - len(inputs)]

                # pad,还加attn_mask嘛
                input_ids = input_ids + ([self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids)))

                mlm_labels = [-1] * self.max_seq_length
                mask_list = [self.tokenizer.mask_token_id] * self.max_seq_length
                mlm_labels[new_start+1: new_end] = token_ids[start+1: end]  # mlm的标签
                input_ids[new_start+1: new_end] = mask_list[new_start+1: new_end]

                anchor = self.tokenizer.convert_tokens_to_string(tokens[start+1: end])
                if anchor in annotation:
                    entity = annotation[anchor]
                    if entity in label_set:
                        anchor_label = label_set[entity] # linked entity
                    else:
                        continue
                else:
                    filtered_anchor += 1
                    continue

                assert len(input_ids) == self.max_seq_length
                assert len(mlm_labels) == self.max_seq_length
                assert len(span_id) == self.max_seq_length

                features.append(InputFeatures(input_ids=input_ids,
                              mlm_labels=mlm_labels,
                              mention_span=span_id,
                              mention_entity=anchor_label,  # 这需要转换成idx表示的
                              description_entity=title_id,  # # 这需要转换成idx表示的
                              ))
        logging.info('the total number of pre-training samples of this file: {}'.format(len(features)))
        return features