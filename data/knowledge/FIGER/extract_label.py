# import json
#
# with open('train.json', 'r', encoding='utf8') as f:
#     examples = []
#     lines = json.load(f)
#     for (i, line) in enumerate(lines):
#         label = line['labels']
#         examples.append(label)
#     d = {}
#     for e in examples:
#         for l in e:
#             if l in d:
#                 d[l] += 1
#             else:
#                 d[l] = 1
#     for k, v in d.items():
#         d[k] = (len(examples) - v) * 1. / v
#
#     label_list = list(d.keys())
#     examples = []
# with open('labels.json', "w") as f:
#     json.dump(label_list, f)


# 统计长度
import json
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
# tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
#
#
# def _read_json(input_file):
#     with open(input_file, 'r', encoding='utf8') as f:
#         return json.load(f)
#
#
# def _create_examples(lines):
#     examples = []
#     for (i, line) in enumerate(lines):
#         guid = i
#         text_a = line['sent']
#         tokens = tokenizer.tokenize(text_a)
#         examples.append(tokens)
#     return examples
#
#
# if __name__ == "__main__":
#     lines = _read_json("train.json")
#     examples = _create_examples(lines)
#     length_list = []
#     for x in examples:
#         length_list.append(len(x))
#     with open('length.figer', 'w', encoding='utf-8') as fw:
#         for x in length_list:
#             fw.write(str(x) + '\n')
#     print("avg length: ", sum(length_list) / len(length_list))






# ==== 随机采样2W条，1%
def _read_json(input_file):
    with open(input_file, 'r', encoding='utf8') as f:
        return json.load(f)


if __name__ == "__main__":
    import random
    r_files = ['dev.json', 'test.json', 'train.json']
    w_files = ['../FIGER_2W/dev.json', '../FIGER_2W/test.json', '../FIGER_2W/train.json']
    for r_file, w_file in zip(r_files, w_files):
        lines = _read_json(r_file)
        random.shuffle(lines)
        if 'train' in r_file:
            lines = lines[:20000]
        elif 'dev' in r_file:
            lines = lines[:1000]
        elif 'test' in r_file:
            pass
        with open(w_file, 'w', encoding='utf-8') as fw:
            json.dump(lines, fw)

