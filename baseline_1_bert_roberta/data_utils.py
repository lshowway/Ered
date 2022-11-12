from __future__ import absolute_import, division, print_function

import csv
import numpy as np
import random
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
    def __init__(self, input_ids, input_mask, segment_ids, start_id=None, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_id = start_id
        
        self.label_id = label_id



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label



class DataProcessor(object):

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

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



class OpenentityProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        label_list = ['entity', 'location', 'time', 'organization', 'object', 'event', 'place', 'person', 'group']
        return label_list

    def _create_examples(self, lines):
        examples = []
        label_list = self.get_labels()
        label_set = set()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0] * len(label_list)
            for item in line['labels']:
                label_set.add(item)
                label[label_list.index(item)] = 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class FigerProcessor(DataProcessor):


    def get_train_examples(self, data_dir, dataset_type=None):
        lines = self._read_json(os.path.join(data_dir, "train.json"))
        return self._create_examples(lines)

    def get_dev_examples(self, data_dir, dataset_type):
        lines = self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type)))
        return self._create_examples(lines)

    def get_labels(self):
        label_list = ["/person/artist", "/person", "/transportation", "/location/cemetery", "/language", "/location",
                      "/location/city", "/transportation/road", "/person/actor", "/person/soldier",
                      "/person/politician", "/location/country", "/geography", "/geography/island", "/people",
                      "/people/ethnicity", "/internet", "/internet/website", "/broadcast_network", "/organization",
                      "/organization/company", "/person/athlete", "/organization/sports_team", "/location/county",
                      "/geography/mountain", "/title", "/person/musician", "/event",
                      "/organization/educational_institution",
                      "/person/author", "/military", "/astral_body", "/written_work", "/event/military_conflict",
                      "/person/engineer",
                      "/event/attack", "/organization/sports_league", "/government", "/government/government",
                      "/location/province",
                      "/chemistry", "/music", "/education/educational_degree", "/education",
                      "/building/sports_facility",
                      "/building", "/government_agency", "/broadcast_program", "/living_thing", "/event/election",
                      "/location/body_of_water", "/person/director", "/park", "/event/sports_event", "/law",
                      "/product/ship", "/product", "/product/weapon", "/building/airport", "/software",
                      "/computer/programming_language",
                      "/computer", "/body_part", "/disease", "/art", "/art/film", "/person/monarch", "/game", "/food",
                      "/person/coach", "/government/political_party", "/news_agency", "/rail/railway", "/rail",
                      "/train",
                      "/play", "/god", "/product/airplane", "/event/natural_disaster", "/time", "/person/architect",
                      "/award", "/medicine/medical_treatment", "/medicine/drug", "/medicine",
                      "/organization/fraternity_sorority",
                      "/event/protest", "/product/computer", "/person/religious_leader", "/religion",
                      "/religion/religion",
                      "/building/theater", "/biology", "/livingthing", "/livingthing/animal", "/finance/currency",
                      "/finance",
                      "/organization/airline", "/product/instrument", "/location/bridge", "/building/restaurant",
                      "/medicine/symptom",
                      "/product/car", "/person/doctor", "/metropolitan_transit", "/metropolitan_transit/transit_line",
                      "/transit",
                      "/product/spacecraft", "/broadcast", "/broadcast/tv_channel", "/building/library",
                      "/education/department", "/building/hospital"]
        return label_list

    def _create_examples(self, lines):
        examples = []
        label_list = self.get_labels()
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['sent']
            text_b = (line['start'], line['end'])
            label = [0] * len(label_list)
            for item in line['labels']:
                label[label_list.index(item)] = 1
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class FewrelProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type):
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "train")
        return examples

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))), "{}".format(dataset_type))

    def get_labels(self):
        labels = ['P22', 'P449', 'P137', 'P57', 'P750', 'P102', 'P127', 'P1346', 'P410', 'P156', 'P26', 'P674', 'P306', 'P931',
         'P1435', 'P495', 'P460', 'P1411', 'P1001', 'P6', 'P413', 'P178', 'P118', 'P276', 'P361', 'P710', 'P155',
         'P740', 'P31', 'P1303', 'P136', 'P974', 'P407', 'P40', 'P39', 'P175', 'P463', 'P527', 'P17', 'P101', 'P800',
         'P3373', 'P2094', 'P135', 'P58', 'P206', 'P1344', 'P27', 'P105', 'P25', 'P1408', 'P3450', 'P84', 'P991',
         'P1877', 'P106', 'P264', 'P355', 'P937', 'P400', 'P177', 'P140', 'P1923', 'P706', 'P123', 'P131', 'P159',
         'P641', 'P412', 'P403', 'P921', 'P176', 'P59', 'P466', 'P241', 'P150', 'P86', 'P4552', 'P551', 'P364']
        return labels

    def _create_examples(self, lines, dataset_type):
        examples = []
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (dataset_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
            text_a = line['text']
            text_b = (line['ents'][0][1], line['ents'][0][2], line['ents'][1][1],  line['ents'][1][2])
            # neighbour = line['ents']
            label = line['label']
            label = label_map[label]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


TACRED_relations = ['per:siblings', 'per:parents', 'org:member_of', 'per:origin', 'per:alternate_names', 'per:date_of_death',
             'per:title', 'org:alternate_names', 'per:countries_of_residence', 'org:stateorprovince_of_headquarters',
             'per:city_of_death', 'per:schools_attended', 'per:employee_of', 'org:members', 'org:dissolved',
             'per:date_of_birth', 'org:number_of_employees/members', 'org:founded', 'org:founded_by',
             'org:political/religious_affiliation', 'org:website', 'org:top_members/employees', 'per:children',
             'per:cities_of_residence', 'per:cause_of_death', 'org:shareholders', 'per:age', 'per:religion',
             'NA',
             'org:parents', 'org:subsidiaries', 'per:country_of_birth', 'per:stateorprovince_of_death',
             'per:city_of_birth',
             'per:stateorprovinces_of_residence', 'org:country_of_headquarters', 'per:other_family',
             'per:stateorprovince_of_birth',
             'per:country_of_death', 'per:charges', 'org:city_of_headquarters', 'per:spouse']



class TACREDProcessor(DataProcessor):

    def get_train_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "{}.json".format(dataset_type))))

    def get_labels(self, ):
        labels = set(TACRED_relations)
        if 'NA' in labels:
            labels.discard("NA")
            return ["NA"] + sorted(labels)
        else:
            return sorted(labels)

    def _create_examples(self, lines, ):
        examples = []
        label_set = self.get_labels()
        label_map = {l: i for i, l in enumerate(label_set)}
        for (i, line) in enumerate(lines):
            guid = i
            text_a = line['text']
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0

            sub = (line["ents"][0][1], line["ents"][0][2])
            obj = (line["ents"][1][1], line["ents"][1][2])

            text_b = (sub[0], sub[1], obj[0], obj[1])

            label = line['label']
            label = label_map[label]
            # neighbour = line['ann']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        random.shuffle(examples)
        return examples



class Sst2Processor(DataProcessor):
    def get_train_examples(self, data_dir, dataset_type=None):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, dataset_type):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset_type))), dataset_type)

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, lines, set_type):
        examples = []
        label_list = {"0": 0, "1": 1}
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i
            text_a, l = line
            label = label_list[l]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def load_and_cache_examples(args, task, tokenizer, dataset_type, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    input_mode = input_modes[task]
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        task,
        dataset_type,
        args.model_name_or_path,
        str(args.max_seq_length),
        ))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        # label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, dataset_type)
        else:
            examples = processor.get_train_examples(args.data_dir, dataset_type)
        if input_mode == 'entity_sentence':
            features = convert_examples_to_features_entity_typing(
                args, examples, args.max_seq_length, tokenizer)
        elif input_mode == "entity_entity_sentence":
            features = convert_examples_to_features_relation_classification(
                examples, args.max_seq_length, tokenizer)
        else:
            features = convert_examples_to_features_single(examples, args.max_seq_length, tokenizer,
                                                           pad_token_segment_id=1 if 'bert' in args.model_type else 0,
                                                           )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if args.task_name in ['openentity', 'figer']:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    if args.task_name in ['sst2']:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    elif args.task_name in ['openentity', 'figer', 'fewrel', 'tacred']:
        all_start_ids = torch.tensor([f.start_id for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_ids, all_label_ids)
    else:
        dataset = None
    return dataset


def convert_examples_to_features_entity_typing(args, examples, origin_seq_length, tokenizer):
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
        tokens = [tokenizer.cls_token] + tokens_0_start + tokenizer.tokenize("[ENTITY]") + tokens_start_end + tokenizer.tokenize("[ENTITY]") + tokens_end_last + [tokenizer.sep_token]
        tokens = tokens[: origin_seq_length]
        start = 1 + len(tokens_0_start)
        end = 1 + len(tokens_0_start) + 1 + len(tokens_start_end)
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding_length = origin_seq_length - len(input_ids)
        # pad
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        # label
        label_id = example.label
        start_id = np.zeros(origin_seq_length)
        if start >= origin_seq_length:
            start = 0  # 如果entity被截断了，就使用CLS位代替
        start_id[start] = 1
        # start_id[end] = 1

        if args.task_name in ['tacred']:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              label_id=label_id,
                              start_id=start_id,

                              ))
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              start_id=start_id,
                              ))
    return features



def convert_examples_to_features_relation_classification(examples,
                                                         origin_seq_length,
                                                         tokenizer,):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # ==== backbone ====
        text_a = example.text_a
        start_0, end_0, start_1, end_1 = example.text_b
        before_sub = text_a[:start_0].strip()
        tokens = [tokenizer.cls_token] + tokenizer.tokenize(before_sub)
        sub_start = len(tokens)
        tokens += ['@']
        sub = text_a[start_0: end_0 + 1].strip()
        tokens += tokenizer.tokenize(sub)
        tokens += ['@']
        sub_end = len(tokens)
        between_sub_obj = text_a[end_0 + 1: start_1].strip()
        tokens += tokenizer.tokenize(between_sub_obj)
        obj_start = len(tokens)
        tokens += ['#']
        obj = text_a[start_1: end_1 + 1].strip()
        tokens += tokenizer.tokenize(obj)
        tokens += ['#']
        obj_end = len(tokens)
        after_obj = text_a[end_1 + 1:].strip()
        tokens += tokenizer.tokenize(after_obj) + [tokenizer.sep_token]

        tokens = tokens[: origin_seq_length]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # pad
        padding_length = origin_seq_length - len(input_ids)
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([tokenizer.pad_token_type_id] * padding_length)
        assert len(input_ids) == origin_seq_length
        assert len(input_mask) == origin_seq_length
        assert len(segment_ids) == origin_seq_length
        # label
        label_id = example.label
        # sure that sub & obj are included in the sequence
        if sub_start > origin_seq_length - 1:
            sub_start = 0
        if obj_start > origin_seq_length - 1:
            obj_start = 0
        if sub_end > origin_seq_length - 1:
            sub_end = origin_seq_length
        if obj_end > origin_seq_length:
            obj_end = origin_seq_length
        # the sub_special_start_id is an array, where the idx of start id is 1, other position is 0.
        subj_special_start_id = np.zeros(origin_seq_length)
        obj_special_start_id = np.zeros(origin_seq_length)
        subj_special_start_id[sub_start] = 1
        obj_special_start_id[obj_start] = 1

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: {}".format(label_id))
            logger.info("start_id: {}".format((sub_start, obj_start)))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          start_id=(subj_special_start_id, obj_special_start_id),
                          label_id=example.label,
                          ))

    return features


def convert_examples_to_features_single(examples, max_seq_length,
                                        tokenizer,
                                        pad_token_segment_id=1,
                                        ):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_a = tokens_a[: max_seq_length - 2]

        tokens = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # Zero-pad
        padding_length = max_seq_length - len(input_ids)
        input_ids += ([tokenizer.pad_token_id] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([pad_token_segment_id] * padding_length)  # 0 0 0 1 1 1 0 0

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label

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
    "sst2": Sst2Processor,
    "openentity": OpenentityProcessor,
    "figer": FigerProcessor,
    "tacred": TACREDProcessor,
    "fewrel": FewrelProcessor,
}

output_modes = {
    "qqp": "classification",
    "qnli": "classification",
    "wnli": "classification",
    "sst2": "classification",
    "eem": "classification",
    "openentity": "classification",
    "figer": "classification",
    "tacred": "classification",
    "fewrel": "classification",
}

input_modes = {
    "qqp": "sentence_pair",
    "qnli": "sentence_pair",
    "wnli": "sentence_pair",
    "sst2": "single_sentence",
    "eem": "sentence_pair",
    "openentity": "entity_sentence",
    "figer": "entity_sentence",
    "tacred": "entity_entity_sentence",
    "fewrel": "entity_entity_sentence",
}
final_metric = {
    'sst2': 'accuracy',
    "eem": 'roc_auc',
    "openentity": 'micro_F1',
    "figer": 'micro_F1',
    "tacred": 'micro_F1',
    "fewrel": 'micro_F1'

}