# -*- coding: utf-8 -*-

import click
import gzip
import os
import rdflib
import re
import urllib
import json
from urllib import parse
from collections import defaultdict
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from shelve import DbfilenameShelf

from tokenizer import RegexpTokenizer


class AbstractDB(DbfilenameShelf):
    def __init__(self, *args, **kwargs):
        DbfilenameShelf.__init__(self, *args, **kwargs)

    @staticmethod
    def build(in_dir, out_file, pool_size):
        with open('G:\D\MSRA\knowledge_aware\knowledge_resource\dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json', 'w') as fw:
            with closing(AbstractDB(out_file, protocol=-1)) as db:
                target_files = [f for f in sorted(os.listdir(in_dir)) if f.endswith('ttl.gz')]
                with closing(Pool(pool_size)) as pool:
                    f = partial(_process_file, in_dir=in_dir)
                    for ret in pool.imap(f, target_files):
                        for x in ret:
                            json.dump(x, fw)
                            fw.write('\n')

    def count_valid_words(self, vocab, max_text_len):
        tokenizer = RegexpTokenizer()
        keys = self.keys()
        words = frozenset(list(vocab.words()))
        word_count = 0

        with click.progressbar(keys) as bar:
            for key in bar:
                c = 0
                for token in tokenizer.tokenize(self[key]['text']):
                    if token.text.lower() in words:
                        c += 1

                word_count += min(c, max_text_len)

        return word_count


def _process_file(file_name, in_dir):
    abs_matcher = re.compile('^http://dbpedia\.org/resource/(.*)/abstract#offset_(\d+)_(\d+)$')  # the start of one entity abstract
    dbp_matcher = re.compile('^http://dbpedia\.org/resource/(.*)$')  # dbpedia's entity
    click.echo('Processing %s' % file_name)
    g = rdflib.Graph()
    with gzip.GzipFile(os.path.join(in_dir, file_name)) as f:
        g.load(f, format='turtle')
    # g.load("G:\D\MSRA\knowledge_aware\knowledge_resource\dbpedia_abstract_corpus/abstracts_en0.ttl", format='turtle')
    # g.load(os.path.join(in_dir, file_name), format='turtle')

    abstract_dict = {}
    mentions_span_dict = defaultdict(list)
    span_entity_dict = defaultdict(list)
    for (s, p, o) in g: # 三元组
        if p == rdflib.URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#isString'): # 此时SPO分别是：global entity的0-len，o是abstract
            abs_match_obj = abs_matcher.match(s)  # 无论是abstract还是mention的都有这个
            t1 = abs_match_obj.group(1)
            global_entity = parse.unquote(parse.unquote(t1))  # global entity
            global_len = int(abs_match_obj.group(3))  # mention span
            abstract_dict[global_entity] = (global_len, str(o))  # o是abstract
        elif p == rdflib.URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf'): # s：mention的span的，o：是mention
            abs_match_obj = abs_matcher.match(s)  # 无论是abstract还是mention的都有这个
            t1 = abs_match_obj.group(1)
            global_entity = parse.unquote(parse.unquote(t1))  # global entity
            span = (int(abs_match_obj.group(2)), int(abs_match_obj.group(3)))  # mention span
            mentions_span_dict[global_entity].append((str(o), span))  # o：entity mention, span: mention span
        elif p == rdflib.URIRef('http://www.w3.org/2005/11/its/rdf#taIdentRef'): # 此时SPO分别是：mention的offset那些，mention的entity那些
            abs_match_obj = abs_matcher.match(s)
            t1 = abs_match_obj.group(1)
            global_entity = parse.unquote(parse.unquote(t1))  # global entity
            span = (int(abs_match_obj.group(2)), int(abs_match_obj.group(3)))  # mention span

            match_obj = dbp_matcher.match(o)
            if match_obj:
                t2 = match_obj.group(1)
                mention_entity = parse.unquote(t2) # mention对应的entity
                span_entity_dict[global_entity].append((span, mention_entity))

    ret = []
    for (global_entity, (global_len, abstract)) in abstract_dict.items():
        if global_entity in mentions_span_dict and global_entity in span_entity_dict:
            mention_span_list = mentions_span_dict[global_entity]
            span_entity_list = span_entity_dict[global_entity]

            t1 = {x[0]: x[1] for x in mention_span_list}
            t2 = {x[0]: x[1] for x in span_entity_list}

            mention_list = []
            for mention, span in t1.items():
                if span in t2:
                    entity = t2[span]
                    mention_list.append((mention, entity, span))

            line_dict = {"global_entity_name": global_entity, "global_len": global_len,
                         "abstract": abstract, "abstract_mentions": mention_list}
        else:
            line_dict = {"global_entity_name": global_entity, "global_len": global_len,
                         "abstract": abstract, "abstract_mentions": []}

        ret.append(line_dict)

    return ret
