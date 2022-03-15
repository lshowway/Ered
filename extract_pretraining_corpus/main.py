# -*- coding: utf-8 -*-
import json
import time

import click
import wget
from wikimapper import WikiMapper
from multiprocessing import Pool

from abstract_db import AbstractDB


def download_dbpedia_abstract_files(out_dir):
    for n in range(114):
        url = 'http://downloads.dbpedia.org/2015-04/ext/nlp/abstracts/en/abstracts_en%d.ttl.gz' % (n,)
        click.echo('Getting %s' % url)
        # subprocess.getstatusoutput('wget -P %s/ %s' % (out_dir, url))  # unix
        # subprocess.getstatusoutput('curl -o %s %s' % (out_dir, url))  # window
        filename = wget.detect_filename(url)
        wget.download(url)



def build_abstract_db(in_dir, out_file, pool_size):
    AbstractDB.build(in_dir, out_file, pool_size)


mapper = WikiMapper("../knowledge_resource/index_enwiki-20190420.db")
def extract_qid(mention_entity):
    mention_entity_qid = mapper.title_to_id(mention_entity)
    if mention_entity_qid is not None:
        return mention_entity, mention_entity_qid
    else:
        return mention_entity, "NO_QID"


def read_json_lines(file):
    all_jsonlines = []
    with open(file) as fr:
        for x in fr:
            x = json.loads(x)
            all_jsonlines.append(x)
            # if len(all_jsonlines) > 10000:
            #     break
    return all_jsonlines


def static_entities_size(file):
    # include entities of both in abstract and title
    entity_set = set()  # 5913023
    with open(file) as fr:
        for x in fr:
            x = json.loads(x)
            global_entity_name = x['global_entity_name']
            entity_set.add(global_entity_name)  # add title
            abstract_mentions_list = x['abstract_mentions']
            for z in abstract_mentions_list:
                mention_entity = z[1]
                entity_set.add(mention_entity)  # add entity in abstract
            # if len(entity_set) > 50000:
            #     break

    return list(entity_set)


def add_qid(all_json_lines, entity_qid_dic, out_file):
    new_all_lines = []
    count = 0
    for x in all_json_lines:
        global_entity_name = x['global_entity_name']
        global_entity_name_qid = entity_qid_dic[global_entity_name]
        if global_entity_name_qid == "NO_QID":
            count += 1
            continue  # 如果这个entity没有qid，那么他的abstract就不使用
        global_len = x['global_len']
        abstract = x['abstract']
        abstract_mentions_list = x['abstract_mentions']
        abstract_mentions_list_new = []
        for z in abstract_mentions_list:
            mention = z[0]
            mention_entity = z[1]
            mention_span = z[2]
            mention_entity_qid = entity_qid_dic[mention_entity]
            if mention_entity_qid == 'NO_QID':
                print(mention_entity)
            else:  # 只保留有qid的entity
                t = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span, "mention_entity_qid": mention_entity_qid}
                abstract_mentions_list_new.append(t)
        line_dict = {"global_entity_name_qid": global_entity_name_qid,
                     "global_entity_name": global_entity_name,
                     "global_len": global_len,
                     "abstract": abstract,
                     "abstract_mentions": abstract_mentions_list_new}
        new_all_lines.append(line_dict)
    print('the number of pages not with qid: ', count)

    with open(out_file, 'w', encoding='utf-8') as fw:
        for x in new_all_lines:
            json.dump(x, fw)
            fw.write('\n')


def read_entity_qid_dic(file):
    entity_qid_dic = {}
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            entity, qid = line.strip().split('\t')
            entity_qid_dic[entity] = qid
    return entity_qid_dic


if __name__ == "__main__":
    # 第一步：下载DBpedia abstract corpus语料
    # 这儿代码没下载到哪个文件夹去，此外这个处理文件需要写成参数形式用CLI
    # download_dbpedia_abstract_files(out_dir='../knowledge_resource/dbpedia_abstract_corpus')
    # 第二步：extract v1
    # download index_enwiki-20190420.db
    # wget https://public.ukp.informatik.tu-darmstadt.de/wikimapper/index_enwiki-20190420.db
    # build_abstract_db(in_dir='../knowledge_resource/dbpedia_abstract_corpus', out_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json", pool_size=100)

    # 第三步：extract QID 592M
    # all_jsonlines = static_entities_size(file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json")
    # file = "../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv"
    # with open(file, 'w', encoding='utf-8') as fw:
    #     span = 10000
    #     max_rn = 0
    #     max_end = 592  # 5913023
    #     for i in range(max_rn, max_end):
    #         start = time.time()
    #         print(i, start)
    #         tmp = all_jsonlines[span * i: span * (i + 1)]
    #         with Pool(100) as p:
    #             t = p.map(extract_qid, tmp)
    #         print(time.time()-start)
    #         for x in t:
    #             fw.write('\t'.join(x))
    #             fw.write('\n')
    #         fw.flush()

    # 第四步：add QID
    all_json_lines = read_json_lines(file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json")
    entity_qid_dic = read_entity_qid_dic('../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv')
    add_qid(all_json_lines, entity_qid_dic, out_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v2.json") # with qid


    # 第四步：所有entity中，去掉单字的，且在vocab.txt中出现的（能出现说明挺频繁的了吧）


