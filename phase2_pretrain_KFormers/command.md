#### the command of pretraining K-module
nohup python run_KFormers.py  --model_type  KModule  --model_type roberta   --model_name_or_path  roberta-large   --max_seq_length 128  --train_batch_size 64   --valid_batch_size 512   --max_steps 1  --epochs 3  --learning_rate 5e-5  --eval_steps -1    --fp16 True   --data_dir ../data/gluedata/SST2  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.SST2.64.5e-5.128.KFormers &

--data_dir, --model_type, --model_name_or_path, --task_name, --bert_model