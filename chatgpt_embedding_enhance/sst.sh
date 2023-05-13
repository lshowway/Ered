#!/bin/bash
#SBATCH --job-name=entered
#SBATCH --ntasks=8 --cpus-per-task=4 --mem=16000M
#SBATCH --time=04:00:00

nvidia-smi

python run_entered_chatgpt_embedding.py  --task_name  sst2     --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  gpt-3.5-turbo  --backbone_seq_length  128  --knowledge_seq_length -1  --qid_file  ../data/knowledge/pretrain/wikipedia_description/wikidata5m_des.wikipedia   --max_ent_num -1  --max_des_num 1  --train_batch_size 64   --valid_batch_size 256   --gradient_accumulation_steps 1   --max_steps -1   --update_K_module False   --data_dir ../data/gluedata/SST2/SST2_tagme  --output_dir may_15   --epochs 3   --learning_rate 1e-5  --eval_steps 50  --warmup_steps -1  --alpha 1.0  --beta 0.001   --seed 42 