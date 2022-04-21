import json
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Dict

import click
import torch

from luke_modeling import LukeConfig
from data_utils import EntityVocab

from transformers import XLMRobertaTokenizer as OriginalXLMRobertaTokenizer
from transformers import AutoTokenizer as OriginalAutoTokenizer


MODEL_FILE = "pytorch_model.bin"
METADATA_FILE = "metadata.json"
TSV_ENTITY_VOCAB_FILE = "entity_vocab.tsv"
ENTITY_VOCAB_FILE = "entity_vocab.jsonl"



def get_entity_vocab_file_path(directory: str) -> str:
    default_entity_vocab_file_path = os.path.join(directory, ENTITY_VOCAB_FILE)
    tsv_entity_vocab_file_path = os.path.join(directory, TSV_ENTITY_VOCAB_FILE)

    if os.path.exists(tsv_entity_vocab_file_path):
        return tsv_entity_vocab_file_path
    elif os.path.exists(default_entity_vocab_file_path):
        return default_entity_vocab_file_path
    else:
        raise FileNotFoundError(f"{directory} does not contain any entity vocab files.")





class ModelArchive(object):
    def __init__(self, state_dict: Dict[str, torch.Tensor], metadata: dict, entity_vocab: EntityVocab):
        self.state_dict = state_dict
        self.metadata = metadata
        self.entity_vocab = entity_vocab

    @property
    def bert_model_name(self):
        return self.metadata["model_config"]["bert_model_name"]

    @property
    def config(self):
        config = LukeConfig(**self.metadata["model_config"])
        if self.bert_model_name.startswith("roberta"):  # for compatibility for recent transformers
            config.pad_token_id = 1
        return config

    @property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.bert_model_name)

    @property
    def max_seq_length(self):
        return self.metadata["max_seq_length"]

    @property
    def max_mention_length(self):
        return self.metadata["max_mention_length"]

    @property
    def max_entity_length(self):
        return self.metadata["max_entity_length"]

    @classmethod
    def load(cls, archive_path: str):
        if os.path.isdir(archive_path):
            return cls._load(archive_path, MODEL_FILE)
        elif archive_path.endswith(".bin"):
            return cls._load(os.path.dirname(archive_path), os.path.basename(archive_path))

        with tempfile.TemporaryDirectory() as temp_path:
            f = tarfile.open(archive_path)
            f.extractall(temp_path)
            return cls._load(temp_path, MODEL_FILE)

    @staticmethod
    def _load(path: str, model_file: str):
        state_dict = torch.load(os.path.join(path, model_file), map_location="cpu")
        with open(os.path.join(path, METADATA_FILE)) as metadata_file:
            metadata = json.load(metadata_file)
        entity_vocab = EntityVocab(get_entity_vocab_file_path(path))

        return ModelArchive(state_dict, metadata, entity_vocab)






class XLMRobertaTokenizer(OriginalXLMRobertaTokenizer):
    """
        The original XLMRobertaTokenizer is broken, so fix that ourselves.
        (https://github.com/huggingface/transformers/issues/2976)
    """

    def __init__(self, vocab_file, **kwargs):
        super().__init__(vocab_file, **kwargs)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + self.fairseq_offset
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    @property
    def vocab_size(self):
        return len(self.sp_model) + self.fairseq_offset


class AutoTokenizer(OriginalAutoTokenizer):
    """
        A wrapper class of transformers.AutoTokenizer.
        This returns our fixed version of XLMRobertaTokenizer in from_pretrained().
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if "xlm-roberta" in pretrained_model_name_or_path:
            return XLMRobertaTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        else:
            return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
