import os
import json


def read_and_split_entities(file, all_abstracts=None, output=None):
    all_qids = []
    all_lines = []
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            all_lines.append(line)
            all_qids.append(qid)

    fenshu = len(all_qids) // 50265 + 1
    for x in range(fenshu):
        this_split_line = all_lines[x * 50265: 50265 * (x+1)]
        this_split_qid = all_qids[x * 50265: 50265 * (x+1)]
        if not os.path.exists(os.path.join(output, str(x))):
            os.makedirs(os.path.join(output, str(x)))
        # 1. write vocab
        with open(os.path.join(output, str(x), 'entity_vocab_%s'%x), 'w', encoding='utf-8') as fw:
            for y in this_split_line:
                fw.write(y)

        # 2. write corpus
        with open(os.path.join(output, str(x), 'pretraining_corpus_%s' % x), 'w', encoding='utf-8') as fw:
            count = 0
            for x in all_abstracts:
                global_entity_qid = x['global_entity_name_qid']
                if global_entity_qid not in this_split_qid:
                    continue
                abstract_mentions_list = x['abstract_mentions']
                for z in abstract_mentions_list:
                    mention_qid = z['mention_entity_qid']
                    if mention_qid not in this_split_qid:
                        continue
                json.dump(x, fw)
                fw.write('\n')
                count += 1
            print(count)



def rewrite_abstract(used_entities, abstract_file, output_file):
    all_lines = []
    with open(abstract_file) as fr:
        count = 0
        for x in fr:
            count += 1
            # if count > 50000:
            #     break
            x = json.loads(x)
            all_lines.append(x)

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
        new_all_lines.append(x)

    with open(output_file, 'w', encoding='utf-8') as fw:
        for x in new_all_lines:
            json.dump(x, fw)
            fw.write('\n')


def read_and_split_entities_v2(file, all_abstracts=None, output=None):
    all_entities= set()
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            all_entities.add(title)

    idx = 0
    this_entities = set()
    this_corpus = []
    fw_1 = open(os.path.join(output, str(idx), 'entity_vocab.tsv'), 'w', encoding='utf-8')
    fw_2 = open(os.path.join(output, str(idx), 'pretraining_corpus.json'), 'w', encoding='utf-8')

    for x in all_abstracts:
        if len(this_entities) == 0:
            if not os.path.exists(os.path.join(output, str(idx))):
                os.makedirs(os.path.join(output, str(idx)))
            # 1. write vocab & corpus
            fw_1 = open(os.path.join(output, str(idx), 'entity_vocab.tsv'), 'w', encoding='utf-8')
            fw_2 = open(os.path.join(output, str(idx), 'pretraining_corpus.json'), 'w', encoding='utf-8')

        this_corpus.append(x)

        global_entity_name = x['global_entity_name']
        if global_entity_name in all_entities: # 这是使用的
            this_entities.add(global_entity_name)
        abstract_mentions_list = x['abstract_mentions']
        for z in abstract_mentions_list:
            mention_entity = z['mention_entity']
            this_entities.add(mention_entity)

        if len(this_entities) > (50000 - 10):
            for y in this_entities:
                fw_1.write(y)
            for y in this_corpus:
                json.dump(y, fw_2)
                fw_2.write('\n')
            this_entities = set()
            this_corpus = []
            idx += 1


def read_used_entity(file):
    all_entities = set()
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            all_entities.add(title)

    return all_entities


if __name__ == "__main__":
    used_entities = read_used_entity('../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v3.tsv')
    rewrite_abstract(used_entities,
                     abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v2.json",
                     output_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v3.json")
    # all_lines = read_abstract()
    # # v3是不在vocab，不在LUKE中的全部，有些没有qid
    # read_and_split_entities_v2(,
    #                         all_lines,
    #                         "../pretraining_corpus")


