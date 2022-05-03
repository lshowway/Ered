all_entities = []
from wikimapper import WikiMapper
mapper = WikiMapper("../knowledge_resource/index_enwiki-20190420.db")
with open('../data/knowledge/entity_qid_vocab.tsv', 'w', encoding='utf-8') as fw:
    count = -1
    with open('G:\D\MSRA\knowledge_aware\Annotated-WikiExtractor-master/annotated_wikiextractor/entity_vocab.tsv', encoding='utf-8') as f:
        for line in f:
            count += 1
            if count % 1000  == 0:
                print(count)
            en = line.strip()
            all_entities.append(en)
            id = mapper.title_to_id(en)
            if id:
                # print(en, id)
                fw.write('\t'.join([en, id, str(count)]) + '\n')
            # else:
            #     fw.write('\t'.join([en, 'None']) + '\n')


