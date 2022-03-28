#source ~/zhaoqh_venv/kformers/bin/activate
#cd /home/LAB/huaym/zhaoqh/KFormers/KFormers_roberta_bert_knowledge

CUDA_VISIBLE_DEVICES='0'    python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256  --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 2e-5  --eval_steps -1  --warmup_steps -1     --task_name  fewrel  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/fewrel  --output_dir FR_14   >  logs/log.KFormers.roberta-large.distilbert.FewRel.32.2e-5.256.noKnowledge.10  


CUDA_VISIBLE_DEVICES='0'    python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256  --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 3e-5  --eval_steps -1  --warmup_steps -1     --task_name  fewrel  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/fewrel  --output_dir FR_14   >  logs/log.KFormers.roberta-large.distilbert.FewRel.32.3e-5.256.noKnowledge.10  

