import ast
from wikimapper import WikiMapper
import requests
API_ENDPOINT = "https://www.wikidata.org/w/api.php"


def read_write_file(r_file, w_file, K=1):
    with open(w_file, 'w', encoding='utf-8') as fw:
        fw.write("sentence" + '\t' + "label" + '\n')
        idx = 0
        with open(r_file, encoding='utf-8') as f:
            next(f)
            for line in f:
                idx += 1
                # if idx < 47500:
                #     continue
                mention_list, mention_offset_list, \
                entity_list, entity_qid_list, score_list, text = line.split('\t')
                sentence, label = ast.literal_eval(text)
                mention_list = ast.literal_eval(mention_list)
                mention_offset_list = ast.literal_eval(mention_offset_list)
                # mention_offset_list = [sentence.find(x) for x in mention_list]
                mention_offset_list = [str(len(sentence[: x[0]].split())) for x in mention_offset_list]
                entity_list = ast.literal_eval(entity_list)  # 这是Wikipedia的entity
                description_list = []
                for x in entity_list:
                    params = {
                        'action': 'wbsearchentities',
                        'format': 'json',
                        'language': 'en',
                        'search': x
                    }
                    r = requests.get(API_ENDPOINT, params=params).json().get('search', "")
                    # print(r)
                    if r:
                        description_list.append(r[0].get("description", ""))
                    else:
                        description_list.append("")
                # score_list = ast.literal_eval(score_list)
                mention_list = mention_list[: K]
                mention_offset_list = mention_offset_list[: K]
                entity_list = entity_list[: K]
                description_list = description_list[: K]
                if len(mention_list) < K:
                    mention_list += [""] * (K - len(mention_list))
                if len(mention_offset_list) < K:
                    mention_offset_list += ["-1"] * (K - len(mention_offset_list))
                if len(entity_list) < K:
                    entity_list += [''] * (K - len(entity_list))
                if len(description_list) < K:
                    description_list += [""] * (K - len(description_list))

                # row: sentence, mention, offset, entry, des, label
                sentence = [sentence]
                if not mention_list:
                    sentence.extend([""])
                sentence.extend(mention_list)
                sentence.extend(mention_offset_list)
                sentence.extend(entity_list)
                sentence.extend(description_list)
                sentence = ' [SEP] '.join(sentence)
                # print(sentence)
                fw.write(sentence + '\t' + label + '\n')
                fw.flush()


if __name__ == "__main__":
    r1 = "../dev_info.tsv"
    r2 = "../train_info.tsv"

    w1 = "dev.tsv"
    w2 = "train.tsv"

    for r, w in zip([r1, r2], [w1, w2]):
        read_write_file(r, w)

