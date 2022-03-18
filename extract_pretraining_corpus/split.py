import os
import json


def read_used_entity(file):
    all_entities = set()
    with open(file, encoding='utf-8') as fr:
        for line in fr:
            title, qid = line.strip('\n').split('\t')
            all_entities.add(title)

    return all_entities


def read_json_lines(file):
    all_jsonlines = []
    count = 0
    with open(file) as fr:
        for x in fr:
            count += 1
            if count > 100000:
                break
            x = json.loads(x)
            all_jsonlines.append(x)
    return all_jsonlines


# def read_and_split_entities_v2(entity_file, abstract_file=None, output=None):
#     all_entities= read_used_entity(entity_file)
#     all_abstracts = read_json_lines(abstract_file)
#
#     idx = 0
#     this_entities = set()
#     this_corpus = []
#     fw_1 = open(os.path.join(output, str(idx), 'entity_vocab.tsv'), 'w', encoding='utf-8')
#     fw_2 = open(os.path.join(output, str(idx), 'pretraining_corpus.json'), 'w', encoding='utf-8')
#
#     for x in all_abstracts:
#         if len(this_entities) == 0:
#             if not os.path.exists(os.path.join(output, str(idx))):
#                 os.makedirs(os.path.join(output, str(idx)))
#             # 1. write vocab & corpus
#             fw_1 = open(os.path.join(output, str(idx), 'entity_vocab.tsv'), 'w', encoding='utf-8')
#             fw_2 = open(os.path.join(output, str(idx), 'pretraining_corpus.json'), 'w', encoding='utf-8')
#
#         this_corpus.append(x)
#
#         global_entity_name = x['global_entity_name']
#         if global_entity_name in all_entities: # 这是使用的
#             this_entities.add(global_entity_name)
#         abstract_mentions_list = x['abstract_mentions']
#         for z in abstract_mentions_list:
#             mention_entity = z['mention_entity']
#             this_entities.add(mention_entity)
#
#         if len(this_entities) > (50000 - 10):
#             for y in this_entities:
#                 fw_1.write(y + '\n')
#             fw_1.flush()
#             for y in this_corpus:
#                 json.dump(y, fw_2)
#                 fw_2.write('\n')
#             fw_2.flush()
#             this_entities = set()
#             this_corpus = []
#             idx += 1
#


def read_and_split_entities(entity_file, abstract_file=None, output=None):
    all_entities = list(read_used_entity(entity_file))
    all_abstracts = read_json_lines(abstract_file)

    fenshu = len(all_entities) // 100000 + 1
    span = 100000
    fenshu = 4
    for i in range(fenshu):
        this_split_e = all_entities[i * span: span * (i+1)]
        if not os.path.exists(os.path.join(output, str(i))):
            os.makedirs(os.path.join(output, str(i)))
        # 1. write vocab
        with open(os.path.join(output, str(i), 'entity_vocab.tsv'), 'w', encoding='utf-8') as fw:
            for y in this_split_e:
                fw.write(y+'\n')

    # 2. write corpus
    count = 0
    for x in all_abstracts:
        count += 1
        global_entity_name = x['global_entity_name']
        for i in range(fenshu):
            this_split_e = all_entities[i * span: span * (i + 1)]
            with open(os.path.join(output, str(i), 'pretraining_corpus.json'), 'a', encoding='utf-8') as fw:

                if global_entity_name in this_split_e:
                    abstract_mentions_list = x['abstract_mentions']
                    new_mention_list = []
                    for z in abstract_mentions_list:
                        mention_entity = z['mention_entity']
                        if mention_entity in this_split_e:
                            new_mention_list.append(z)
                    if not new_mention_list:
                        continue
                    x['abstract_mentions'] = new_mention_list
                    json.dump(x, fw)
                    fw.write('\n')



if __name__ == "__main__":
    # read_and_split_entities_v2(entity_file='../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v3.tsv',
    #                         abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v3.json",
    #                         output="../pretraining_corpus")

    read_and_split_entities(entity_file='../knowledge_resource/dbpedia_abstract_corpus/entity_qid_v3.tsv',
                            abstract_file="../knowledge_resource/dbpedia_abstract_corpus/our_dbpedia_abstract_corpus_v3.json",
                            output="../pretraining_corpus")