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
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====

        text_a, entity_label = example.text_a, example.entity_label
        text_a = text_a[: max_num_tokens]

        input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=text_a, token_ids_1=None)
        segment_ids = [sequence_a_segment_id] * len(input_ids)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
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


        if ex_index < 10:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in text_a]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("entity label: {}".format(entity_label))
        # ==== backbone ====

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=entity_label
                          ))

    return features



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, entity_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.entity_label = entity_label



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
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(lines, self.tokenizer)

    def get_dev_examples(self, data_dir, dataset_type=None):
        lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset_type)))
        return self._create_examples(lines, self.tokenizer)

    def get_labels(self):
        lines = self._read_tsv(os.path.join(self.data_dir, "all_wikidata5m_QIDs_name.txt"))
        labels = list(set([x[0] for x in lines]))
        return labels

    def _create_examples(self, lines, tokenizer, tokenizing_batch_size=32768):
        examples = []
        batch_input, label_list, first_index = [], [], 0
        label_set = {k: v for v, k in enumerate(self.get_labels())}
        for (i, line) in enumerate(lines):
            first_index = i
            # if i == 0:
            #     continue
            qid, name, description = line
            label = label_set[qid]

            label_list.append(label)
            batch_input.append(description)

            if len(batch_input) >= tokenizing_batch_size:
                tokenized_input = tokenizer.batch_encode_plus(batch_input, add_special_tokens=False)
                t = tokenized_input['input_ids']  # 存在[]
                for j in range(len(label_list)):
                    examples.append(InputExample(guid=first_index, text_a=t[j], text_b=None, entity_label=label_list[j]))
                batch_input, label_list = [], []
        if len(batch_input) > 0:
            tokenized_input = tokenizer.batch_encode_plus(batch_input, add_special_tokens=False)
            t = tokenized_input['input_ids']  # 存在[]
            for j in range(len(label_list)):
                examples.append(InputExample(guid=first_index, text_a=t[j], text_b=None, entity_label=label_list[j]))
        logger.info(f"Finish creating of size {first_index+1}")
        return examples

