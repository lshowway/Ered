# -*- coding: utf-8 -*-
import json
import time

import click
import wget
from collections import defaultdict
from wikimapper import WikiMapper
from multiprocessing import Pool
from operator import itemgetter




def download_dbpedia_abstract_files(out_dir):
    for n in range(114):
        url = 'http://downloads.dbpedia.org/2015-04/ext/nlp/abstracts/en/abstracts_en%d.ttl.gz' % (n,)
        click.echo('Getting %s' % url)
        # subprocess.getstatusoutput('wget -P %s/ %s' % (out_dir, url))  # unix
        # subprocess.getstatusoutput('curl -o %s %s' % (out_dir, url))  # window
        filename = wget.detect_filename(url)
        wget.download(url)


def build_abstract_db(in_dir, out_file, pool_size):
    # extract information from official NIF file
    from abstract_db import AbstractDB
    AbstractDB.build(in_dir, out_file, pool_size)


mapper = WikiMapper("../knowledge_resource/index_enwiki-20190420.db")
def map_qid(mention_entity):
    mention_entity_qid = mapper.title_to_id(mention_entity)
    if mention_entity_qid is not None:
        return mention_entity, mention_entity_qid
    else:
        return mention_entity, "NO_QID"


def extract_qids(abstract_file, output_entity_file):
    # include entities of both in abstract and title
    entity_set = set()  # 5913023
    with open(abstract_file) as fr:
        for x in fr:
            x = json.loads(x)
            global_entity_name = x['global_entity_name']
            entity_set.add(global_entity_name)  # add title
            abstract_mentions_list = x['abstract_mentions']
            for z in abstract_mentions_list:
                mention_entity = z[1]
                entity_set.add(mention_entity)  # add entity in abstract

    all_jsonlines = list(entity_set)

    with open(output_entity_file, 'w', encoding='utf-8') as fw:
        span = 10000
        max_rn = 0
        max_end = 592  # 5913023
        for i in range(max_rn, max_end):
            start = time.time()
            print(i, start)
            tmp = all_jsonlines[span * i: span * (i + 1)]
            with Pool(100) as p:
                t = p.map(map_qid, tmp)
            print(time.time() - start)
            for x in t:
                fw.write('\t'.join(x))
                fw.write('\n')
            fw.flush()

    return list(entity_set)


def read_entity_qid_dic(file):
    entity_qid_dic = {}
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            entity, qid = line.strip('\n').split('\t')
            if entity:
                entity_qid_dic[entity] = qid
    return entity_qid_dic


def read_json_lines(file):
    all_jsonlines = []
    count = 0
    with open(file) as fr:
        for x in fr:
            count += 1
            # if count > 10000:
            #     break
            x = json.loads(x)
            all_jsonlines.append(x)
    return all_jsonlines


def add_qid(all_json_lines, entity_qid_dic, out_file):
    new_all_lines = []
    for x in all_json_lines:
        global_entity_name = x['global_entity_name']
        if global_entity_name in entity_qid_dic:
            global_entity_name_qid = entity_qid_dic[global_entity_name]
        else:
            continue
        global_len = x['global_len']
        abstract = x['abstract']
        abstract_mentions_list = x['abstract_mentions']
        abstract_mentions_list_new = []
        for z in abstract_mentions_list:
            mention = z[0]
            mention_entity = z[1]
            mention_span = z[2]
            if mention_entity in entity_qid_dic:
                mention_entity_qid = entity_qid_dic[mention_entity]
                t = {"mention": mention, "mention_entity": mention_entity, "mention_span": mention_span, "mention_entity_qid": mention_entity_qid}
                abstract_mentions_list_new.append(t)
            else:
                continue
        line_dict = {"global_entity_name_qid": global_entity_name_qid,
                     "global_entity_name": global_entity_name,
                     "global_len": global_len,
                     "abstract": abstract,
                     "abstract_mentions": abstract_mentions_list_new}
        new_all_lines.append(line_dict)

    with open(out_file, 'w', encoding='utf-8') as fw:
        for x in new_all_lines:
            json.dump(x, fw)
            fw.write('\n')


def remove_entity_in_roberta_large_vocab(roberta_large_vocab, entity_qid, output_entity_qid_remove_roberta):
    roberta_large_vocab = json.loads(open(roberta_large_vocab, encoding='utf-8').read())
    roberta_large_vocab = {k.replace('Ġ', ''): v for k, v in roberta_large_vocab.items()}
    entity_qid_dic = read_entity_qid_dic(entity_qid)
    new_entity_qid_dic = {}  # 5897608
    for k, v in entity_qid_dic.items():
        if k in roberta_large_vocab or k.lower() in roberta_large_vocab:
            continue
        else:
            new_entity_qid_dic[k] = v

    with open(output_entity_qid_remove_roberta, 'w', encoding='utf-8') as fw:
        for k, v in new_entity_qid_dic.items():
            fw.write('\t'.join([k, v]))
            fw.write('\n')


def remove_entity_in_LUKE_vocab(LUKE_vocab_file, entity_vocab_file, output_entity_vocab):
    # LUKE vocab
    LUKE_vocab = set()
    with open(LUKE_vocab_file, encoding='utf-8') as fr:
        next(fr)
        next(fr)
        next(fr)
        count = 0
        for x in fr:
            count += 1
            x = json.loads(x)
            entity = x['entities'][0][0]
            entity = entity.replace(' ', '_')
            LUKE_vocab.add(entity)

    # v2 entity vocab
    entity_qid_dic = {}
    with open(entity_vocab_file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            entity_qid_dic[title] = qid

    # filter 4945967
    filtered = {entity: qid for entity, qid in entity_qid_dic.items() if entity not in LUKE_vocab}

    common = {entity: qid for entity, qid in entity_qid_dic.items() if entity in LUKE_vocab} # 951641

    with open(output_entity_vocab, 'w', encoding='utf-8') as fw:
        for e, qid in filtered.items():
            fw.write('\t'.join([e, qid]) + '\n')


def read_used_entity(file):
    all_entities = set()
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            all_entities.add(title)

    return all_entities


def rewrite_abstract(used_entities, abstract_file, output_file):
    all_lines = read_json_lines(abstract_file)

    new_all_lines = []
    for x in all_lines:
        global_entity_name = x['global_entity_name']
        if global_entity_name not in used_entities:
            continue
        abstract_mentions_list = x['abstract_mentions']

        new_abstract_mentions_list = []
        for z in abstract_mentions_list:
            mention_entity = z['mention_entity']
            if mention_entity in used_entities:
                new_abstract_mentions_list.append(z)
        new_all_lines.append(x) # 329w

    newlist = sorted(new_all_lines, key=itemgetter('global_len'), reverse=True)
    with open(output_file, 'w', encoding='utf-8') as fw:
        for x in newlist:
            json.dump(x, fw)
            fw.write('\n')
            fw.flush()


def discrite_abstract(abstract_file, output_file):
    raw_abstracts = read_json_lines(abstract_file)
    new_abstracts = []
    count = 0
    for x in raw_abstracts:
        global_entity_name_qid = x['global_entity_name_qid']
        global_entity_name = x['global_entity_name']
        global_len = x['global_len']
        abstract = x['abstract']
        abstract_mentions_list = x['abstract_mentions']

        if not abstract_mentions_list:
            count += 1
        for i in range(len(abstract_mentions_list)):
            line_dict = {}

            line_dict['global_entity_name_qid'] = global_entity_name_qid
            line_dict['global_entity_name'] = global_entity_name
            line_dict['global_len'] = global_len
            line_dict['abstract'] = abstract
            line_dict['abstract_mentions'] = [abstract_mentions_list[i]]
            new_abstracts.append(line_dict) #

    with open(output_file, 'w', encoding='utf-8') as fw:
        for x in new_abstracts:
            abstract = x["abstract"]
            global_len = x['global_len']
            if global_len > 2000:  # 修剪一下字符大于2K个的
                start, end = x["abstract_mentions"][0]["mention_span"]
                span = end - start

                if start > 700:
                    abstract = abstract[start-700: end+700]
                    start = 700
                    end = 700 + span
                else:
                    abstract = abstract[: end+1000]
                global_len = len(abstract)
                x["abstract_mentions"][0]["mention_span"] = [start, end]
                x["abstract"] = abstract
                x['global_len'] = global_len
            json.dump(x, fw)
            fw.write('\n')
            fw.flush()
    print(count)


if __name__ == "__main__":
    # 1. download DBpedia abstract corpus; download index_enwiki-20190420.db; extract v1: entity, abstract, mention_list
    # download_dbpedia_abstract_files(out_dir='../knowledge_resource/dbpedia_abstract_corpus')
    # wget https://public.ukp.informatik.tu-darmstadt.de/wikimapper/index_enwiki-20190420.db
    # build_abstract_db(in_dir='../knowledge_resource/dbpedia_abstract_corpus', out_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json", pool_size=100)


    # 2. extract QID 592M
    # extract_qids(abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json",
    #              output_entity_file="../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv")

    # 3. add QID, result in 5.9 million entities
    # entity_qid_dic = read_entity_qid_dic('../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv')
    # all_json_lines = read_json_lines(file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v1.json")
    # add_qid(all_json_lines, entity_qid_dic, out_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v2.json") # with qid


    # 4. remove entity in entity_qid.tsv, which is also in vocab.txt
    remove_entity_in_roberta_large_vocab(roberta_large_vocab='../knowledge_resource/vocab.json',
                                         entity_qid='../knowledge_resource/dbpedia_abstract_corpus/entity_qid.tsv',
                                         output_entity_qid_remove_roberta='../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v2.tsv')


    # 5. remove entity in LUKE vocab
    # remove_entity_in_LUKE_vocab(LUKE_vocab_file="../knowledge_resource/output_ent-vocab.jsonl",
    #                             entity_vocab_file='../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v2.tsv',
    #                             output_entity_vocab='../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v3.tsv')

    # 6. rewrite abstract file, v3, only filtered entities are kept
    used_entities = read_used_entity('../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v3.tsv')
    rewrite_abstract(used_entities,
                     abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v2.json",
                     output_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v3.json")

    # 7. 分散abstract，一个abstract只包括一个entity
    discrite_abstract(abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v3.json",
                      output_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v4.json")
