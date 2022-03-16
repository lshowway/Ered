# -*- coding: utf-8 -*-
import json
import time

import click
import wget
from wikimapper import WikiMapper
from multiprocessing import Pool

from abstract_db import AbstractDB



mapper = WikiMapper("../knowledge_resource/index_enwiki-20190420.db")
def add_qid(mention_entity):
    mention_entity_qid = mapper.title_to_id(mention_entity)
    if mention_entity_qid is not None:
        return mention_entity, mention_entity_qid
    else:
        return mention_entity, "NO_QID"


def static_entities_size(file):
    entity_set = set()  # 5913023
    with open(file) as fr:
        for x in fr:
            x = json.loads(x)
            global_entity_name = x['global_entity_name']
            entity_set.add(global_entity_name)
            abstract_mentions_list = x['abstract_mentions']
            for z in abstract_mentions_list:
                mention_entity = z[1]
                entity_set.add(mention_entity)
            # if len(entity_set) > 50000:
            #     break

    return list(entity_set)


if __name__ == "__main__":

    all_jsonlines = static_entities_size(file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json")
    file = "../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv"
    with open(file, 'w', encoding='utf-8') as fw:
        span = 10000
        max_rn = 0
        max_end = 592  # 5913023
        results = []
        for i in range(max_rn, max_end):
            start = time.time()
            print(i, start)
            tmp = all_jsonlines[span * i: span * (i + 1)]
            with Pool(100) as p:
                t = p.map(add_qid, tmp)
            print(time.time()-start)
            # results.extend(x)
            for x in t:
                fw.write('\t'.join(x))
                fw.write('\n')
            fw.flush()


    # 第四步：所有entity中，去掉单字的，且在vocab.txt中出现的（能出现说明挺频繁的了吧）


