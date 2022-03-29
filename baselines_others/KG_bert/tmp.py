# import csv
# fw = open('../data/qualityControl/EntityPane_hh/qc_entity.set', 'w', encoding='utf-8')
# entity_set = set()
# with open("../data/qualityControl/EntityPane_hh/QC_query_entry_des.tsv", encoding='utf-8') as reader:
#     tsv_reader = csv.reader(reader, delimiter='\t')
#     for record in tsv_reader:
#         id, label, query, keyword, domain, source, \
#         query_mention, query_offset, query_entryName, query_entryDescription, \
#         key_mention, key_offset, key_entryName, key_entryDescription = record
#         entity_set.add(query_entryName)
#         entity_set.add(key_entryName)
# fw.write('\n'.join(list(entity_set)))



# r1 = "../data/CKGv1.3/entity2id.txt"
# r2 = "../data/CKGv1.3/EntityID_name.txt"
# w1 = "../data/CKGv1.3/id_name.dic"
# idx_id = {}
# with open(r1, encoding='utf-8') as f:
#     for line in f:
#         line = line.strip().split('\t')
#         if len(line) != 2:
#             continue
#         idx, id = line
#         idx_id[idx] = id
#
# idx_name = {}
# with open(r2, encoding='utf-8') as f:
#     for line in f:
#         idx, name = line.strip().split('\t')
#         idx_name[idx] = name
# # id_name = {}
# with open(w1, 'w', encoding='utf-8') as fw:
#     for idx, id in idx_id.items():
#         if idx in idx_name:
#             name = idx_name[idx]
#             # id_name[id] = name
#             fw.write(id + '\t' + name + '\n')




# r1 = "../data/CKGv1.3/id_name.dic"
# r2 = '../data/qualityControl/EntityPane_hh/qc_entity.set'
# w1 = "../data/CKGv1.3/name_id.dic.qc"
# id_name = {}
# with open(r1, encoding='utf-8') as f:
#     for line in f:
#         id, name = line.strip().split('\t')
#         id_name[id] = name
# name_id = {v: k for k, v in id_name.items()}
# entities = set()
# with open(r2, encoding='utf-8') as f:
#     for line in f:
#         entity = line.strip()
#         entities.add(entity)
# qc_name_id = {}
# with open(w1, 'w', encoding='utf-8') as fw:
#     for name in entities:
#         if name in name_id:
#             qc_name_id[name] = name_id[name]
#         else:
#             qc_name_id[name] = -1
#     for name, id in qc_name_id.items():
#             fw.write(name+ '\t' + str(id) + '\n')





import numpy as np
r1 = "../data/CKGv1.3/name_id.dic.qc"
r2 = "../data/CKGv1.3/entity2vec.vec"
w1 = '../data/CKGv1.3/name_vec.vec'
name_id = {}
with open(r1, encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        if len(line) == 1:
            name = ''
            id = line[0]
        else:
            name, id = line
        name_id[name] = id
vectors = open(r2, encoding='utf-8').readlines()
with open(w1, 'w', encoding='utf-8') as fw:
    t = list(np.random.normal(0, 0.02, 80))
    t = list(map(str, t))
    s = '\t'.join(t)
    for name, id in name_id.items():
        id = int(id)
        if 0 <= id < len(vectors):
            fw.write(name + '\t' + vectors[id])
        else:
            fw.write(name + '\t' + s +'\n')





