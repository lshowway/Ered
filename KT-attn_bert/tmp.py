from transformers.models.roberta.tokenization_roberta import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
t1 = tokenizer.tokenize('dropped out of New York University to become a copy boy at the New York Herald Tribune')
t2 = tokenizer.tokenize(' dropped out of New York University to become a copy boy at the New York Herald Tribune')
t3 = tokenizer.tokenize('he dropped out of New York University to become a copy boy at the New York Herald Tribune')

tt1 = tokenizer.convert_tokens_to_ids(t1)
tt2 = tokenizer.convert_tokens_to_ids(t2)

print(t1)
print(t2)
print(t3)