import sys
sys.path.append("..")

import logging
import json
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict

from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool


from wikipedia2vec.dump_db import DumpDB
from transformers import RobertaTokenizer


logger = logging.getLogger(__name__)


HEAD_TOKEN = "[HEAD]"
TAIL_TOKEN = "[TAIL]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

Entity = namedtuple("Entity", ["title", "language"])



class InputExample(object):
    def __init__(self, id_, text, span_a, span_b, entities, label):
        self.id = id_
        self.text = text
        self.span_a = span_a
        self.span_b = span_b
        self.entities = entities

        self.label = label



class InputFeatures(object):
    def __init__(
        self,
        word_ids,
        # word_segment_ids,
        word_attention_mask,
        entity_ids,
        entity_position_ids,
        # entity_segment_ids,
        entity_attention_mask,
        k_entity_ids,
        k_entity_mask,
        labels,
    ):
        self.word_ids = word_ids
        # self.word_segment_ids = word_segment_ids
        self.word_attention_mask = word_attention_mask
        self.entity_ids = entity_ids
        self.entity_position_ids = entity_position_ids
        # self.entity_segment_ids = entity_segment_ids
        self.entity_attention_mask = entity_attention_mask

        self.k_entity_ids = k_entity_ids
        self.k_entity_mask = k_entity_mask

        self.labels = labels



class DatasetProcessor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(data_dir, "test")

    def get_label_list(self, data_dir):
        labels = set()
        for example in self.get_train_examples(data_dir):
            labels.add(example.label)
        labels.discard("no_relation")
        return ["no_relation"] + sorted(labels)

    def _create_examples(self, data_dir, set_type):
        with open(os.path.join(data_dir, set_type + ".json"), "r") as f:
            data = json.load(f)

        examples = []
        for i, item in enumerate(data):
            tokens = item["text"]
            token_spans = dict(
                subj=(item["ents"][0][1], item["ents"][0][2]), obj=(item["ents"][1][1], item["ents"][1][2])
            )

            if token_spans["subj"][0] < token_spans["obj"][0]:
                entity_order = ("subj", "obj")
            else:
                entity_order = ("obj", "subj")

            text = ""
            cur = 0
            char_spans = dict(subj=[None, None], obj=[None, None])
            for target_entity in entity_order:
                token_span = token_spans[target_entity]
                text += " ".join(tokens[cur : token_span[0]])
                if text:
                    text += " "
                char_spans[target_entity][0] = len(text)
                text += " ".join(tokens[token_span[0] : token_span[1]]) + " "
                char_spans[target_entity][1] = len(text)
                cur = token_span[1]
            text += " ".join(tokens[cur:])
            text = text.rstrip()

            examples.append(
                InputExample(
                    "%s-%s" % (set_type, i),
                    text,
                    char_spans["subj"],
                    char_spans["obj"],
                    entities=item["ents"],
                    label=item["label"],
                )
            )

        return examples


def convert_examples_to_features(examples, label_list, tokenizer, entity_vocab, max_mention_length, max_ent_num):
    label_map = {l: i for i, l in enumerate(label_list)}

    def tokenize(text):
        text = text.rstrip()
        if isinstance(tokenizer, RobertaTokenizer):
            return tokenizer.tokenize(text, add_prefix_space=True)
        else:
            return tokenizer.tokenize(text)

    features = []
    for example in tqdm(examples):
        if example.span_a[1] < example.span_b[1]:
            span_order = ("span_a", "span_b")
        else:
            span_order = ("span_b", "span_a")
        # 处理word
        tokens = [tokenizer.cls_token]
        cur = 0
        token_spans = {}
        for span_name in span_order:
            span = getattr(example, span_name)
            tokens += tokenize(example.text[cur : span[0]])
            start = len(tokens)
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            tokens += tokenize(example.text[span[0] : span[1]])
            tokens.append(HEAD_TOKEN if span_name == "span_a" else TAIL_TOKEN)
            token_spans[span_name] = (start, len(tokens))
            cur = span[1]

        tokens += tokenize(example.text[cur:])
        tokens.append(tokenizer.sep_token)

        tokens = tokens[:256]

        word_ids = tokenizer.convert_tokens_to_ids(tokens)
        word_attention_mask = [1] * len(tokens)
        word_segment_ids = [0] * len(tokens)
        # 处理entity identifier
        entity_ids = [1, 2]
        entity_position_ids = []
        for span_name in ("span_a", "span_b"):
            span = token_spans[span_name]
            start = span[0]
            end = span[1]
            if start >= len(tokens):
                start = len(tokens) - 1
            if end >= len(tokens):
                end = len(tokens) - 1
            position_ids = list(range(start, end))[:max_mention_length]
            position_ids += [-1] * (max_mention_length - end + start)
            entity_position_ids.append(position_ids)
        entity_segment_ids = [0, 0]
        entity_attention_mask = [1, 1]
        # 处理knowledge：entity
        k_ent_ids = [entity_vocab[ent[0]] for ent in example.entities if ent[0] in entity_vocab]
        k_ent_ids = k_ent_ids[: max_ent_num]
        k_ent_mask = [1] * len(k_ent_ids)
        k_ent_ids += [entity_vocab['[UNK]']] * (max_ent_num - len(k_ent_ids))
        k_ent_mask += [0] * (max_ent_num - len(k_ent_mask))
        # 处理label
        labels = label_map[example.label]
        features.append(
            InputFeatures(
                word_ids=word_ids,
                # word_segment_ids=word_segment_ids,
                word_attention_mask=word_attention_mask,
                entity_ids=entity_ids,
                entity_position_ids=entity_position_ids,
                # entity_segment_ids=entity_segment_ids,
                entity_attention_mask=entity_attention_mask,
                k_entity_ids=k_ent_ids,
                k_entity_mask=k_ent_mask,
                labels=labels, # 单个label
            )
        )

    return features




def load_examples(args, fold="train"):
    if args.local_rank not in (-1, 0) and fold == "train":
        torch.distributed.barrier()

    processor = DatasetProcessor()
    if fold == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif fold == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    label_list = processor.get_label_list(args.data_dir)

    logger.info("Creating features from the dataset...")
    features = convert_examples_to_features(examples, label_list, args.tokenizer, args.max_mention_length)

    if args.local_rank == 0 and fold == "train":
        torch.distributed.barrier()

    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        return dict(
            word_ids=create_padded_sequence("word_ids", args.tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            # word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            # entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            k_entity_ids=create_padded_sequence("k_entity_ids", args.entity_vocab['[UNK]']),
            k_entity_mask=create_padded_sequence("k_entity_mask", 0),
            label=torch.tensor([o.label for o in batch], dtype=torch.long),
        )

    if fold in ("dev", "test"):
        dataloader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        if args.local_rank == -1:
            sampler = RandomSampler(features)
        else:
            sampler = DistributedSampler(features)
        dataloader = DataLoader(features, sampler=sampler, batch_size=args.train_batch_size, collate_fn=collate_fn)

    return dataloader, examples, features, label_list



# ===================
class EntityVocab(object):
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            self._parse_tsv_vocab_file(vocab_file)
        else:
            self._parse_jsonl_vocab_file(vocab_file)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def save(self, out_file: str):
        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
        dump_db: DumpDB,
        out_file: str,
        vocab_size: int,
        white_list: List[str],
        white_list_only: bool,
        pool_size: int,
        chunk_size: int,
        language: str,
    ):
        counter = Counter()
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(Pool(pool_size, initializer=EntityVocab._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocab._count_entities, dump_db.titles(), chunksize=chunk_size):
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title in white_list:
            if counter[title] != 0:
                title_dict[title] = counter[title]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title, count in counter.most_common():
                if title in valid_titles and not title.startswith("Category:"):
                    title_dict[title] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for ent_id, (title, count) in enumerate(title_dict.items()):
                json.dump({"id": ent_id, "entities": [[title, language]], "count": count}, f)
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter
