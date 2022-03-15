#### the command of pretraining K-module
> --warmup_steps -1表示rate=0.1
nohup python run_pretrain_kformers.py   --do_train  --model_type roberta-base  --model_name_or_path  roberta-base  --data_dir  ../data/knowledge/pretrain/wikidata_description  --max_seq_length 32  --num_neg_sample 512  --entity_emb_size 128  --per_gpu_train_batch_size 128  --per_gpu_eval_batch_size 512  --gradient_accumulation_steps 1  --learning_rate 3e-5  --weight_decay 0.0  --num_train_epochs 2  --max_steps -1  --warmup_steps -1  --max_save_checkpoints 10 --save_steps 200  --eval_steps -1  --logging_steps  1000  --seed  3407  --fp16  > Kmodule.post-train.2.24 &

# 参数搜索
1. num_neg_sample
2. lr
3. ws
4. eval_steps应该远大于logging_step，后者是train loss