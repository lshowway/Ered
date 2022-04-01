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

        features =  convert_examples_to_features(examples, args.max_seq_length, tokenizer)
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)
            logger.warning("===> Saving features into cached file %s", cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)

    mention_span = torch.tensor([f.mention_span for f in features], dtype=torch.float)
    mention_entity = torch.tensor([f.mention_entity for f in features], dtype=torch.long)
    description_entity = torch.tensor([f.description_entity for f in features], dtype=torch.long)


    # dataset = TensorDataset(input_ids, input_mask, segment_ids, mlm_labels,
    #                         mention_span, mention_entity, description_entity)

    dataset = TensorDataset(input_ids, input_mask, mlm_labels,
                            mention_span, mention_entity, description_entity)

    return dataset



def convert_examples_to_features(examples, max_seq_length, tokenizer,):
    features = []

    for (ex_index, x) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if ex_index > 10000:
            break

        qid, description_entity, description = x.qid, x.entity_name, x.description
        des_mentions_list = x.des_mentions

        # t = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span,
        #      "mention_entity_qid": mention_entity_qid}
        for y in des_mentions_list:
            mention = y['mention']
            mention_entity = y['mention_entity']
            # mention_qid = y['mention_entity_qid']  # 有NO_QID
            start, end = y['mention_span']
            # text
            before_mention = description[: start]
            this_mention = description[start: end]
            assert this_mention == mention
            after_mention = description[end:]
            # ------
            tokens = [tokenizer.cls_token]

            tokens += tokenizer.tokenize(before_mention)
            new_start = len(tokens)

            tokens += ['@'] # ['SOM']不另设特殊符号，因为需要扩vocab 155
            mention_tokens = tokenizer.tokenize(this_mention)
            tokens += [tokenizer.mask_token] * len(mention_tokens)
            new_end = len(tokens) # 158
            tokens += ['@'] # ['EOM']

            tokens += tokenizer.tokenize(after_mention)

            tokens += [tokenizer.eos_token] # 可能超过，可能不足
            # ========
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            mention_tokens_ids = tokenizer.convert_tokens_to_ids(mention_tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)

            # 还可以是
            # span_id[new_start: new_end+1] = 1  # 中间token也使用
            # === cut
            input_ids = input_ids[: max_seq_length]
            input_mask = input_mask[: max_seq_length]
            segment_ids = segment_ids[: max_seq_length]
            # mlm_labels = mlm_labels[: max_seq_length]
            # span_id = span_id[: max_seq_length]
            # ==
            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            input_mask += [0] * padding_length
            # roberta只有一种token type embedding
            segment_ids += [tokenizer.pad_token_type_id] * padding_length
            # mlm_labels += [-1] * padding_length
            # span_id += [0] * padding_length
            # ===
            # 如果mention被cut了
            span_id = np.zeros(len(input_ids))
            mlm_labels = [-1] * max_seq_length  # pad之后的

            if new_start >= max_seq_length:
                # new_start = new_end = 0
                span_id[0] = 1
                mlm_labels[0] = mention_tokens_ids[0]  # 不包括@@符号
            elif new_end >= max_seq_length:
                new_end = -1  # 如果后面超了，就用最后一个
                span_id[new_start] = 1
                span_id[-1] = 1

                mlm_labels[new_start + 1: max_seq_length] = mention_tokens_ids[: max_seq_length - new_start - 1]  # 不包括@@符号
            else:
                span_id[new_start] = 1
                span_id[new_end] = 1
                mlm_labels[new_start+1: new_end] = mention_tokens_ids

            if sum(mlm_labels) == 0 or sum(span_id) == 0:
                print(new_start, new_end)


                # print(len(mlm_labels[new_start + 1: max_seq_length]), len(mention_tokens_ids[: max_seq_length - new_start - 1]))
            # print(len(mlm_labels))

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(mlm_labels) == max_seq_length
            assert len(span_id) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              mlm_labels=mlm_labels,
                              mention_span = span_id,
                              mention_entity=mention_entity, # 这需要转换成idx表示的
                              description_entity=description_entity, # # 这需要转换成idx表示的
                              # entity_labels=None
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
    def __init__(self, input_ids=None, input_mask=None, segment_ids=None, mlm_labels=None,
                 mention_span=None, mention_entity=None, description_entity=None, entity_labels=None,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
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
    def __init__(self, data_dir=None, entity_vocab_file=None):
        self.entity_vocab_file = entity_vocab_file
        self.data_dir = data_dir

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "our_dbpedia_abstract_corpus_v4.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "our_dbpedia_abstract_corpus_v4_dev.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        entity_vocab = []
        with open(self.entity_vocab_file, encoding='utf-8') as fr:
            for line in fr:
                title, qid = line.strip('\n').split('\t')
                entity_vocab.append(title)
        entity_vocab = dict(zip(entity_vocab, range(len(entity_vocab))))
        return entity_vocab

    def _create_examples(self, lines):
        examples = []
        label_set = self.get_labels() # 4.94 million

        for (i, x) in enumerate(lines):
            qid, entity_name, description, des_mentions_list = \
                x['global_entity_name_qid'], x['global_entity_name'], x['abstract'], x['abstract_mentions']

            # des_mentions = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span,
            #      "mention_entity_qid": mention_entity_qid}
            new_de_mentions_list = []
            for des_mentions in des_mentions_list:
                entity_name = label_set[entity_name] # 不可以报错
                t = des_mentions['mention_entity']
                des_mentions['mention_entity'] = label_set[t]  # 这儿为什么会报错？mention_entity不是根据vocab过滤，或者vocab是根据entity生成的呀？
                new_de_mentions_list.append(des_mentions)
            examples.append(
                InputExample(guid=i, qid=qid, entity_name=entity_name, description=description, des_mentions=new_de_mentions_list))


        return examples