#### the command of pretraining K-module


nohup python -m torch.distributed.launch   --nproc_per_node=4    run_pretrain_kformers.py   --do_train  --model_type roberta  --model_name_or_path  roberta-large  --data_dir  ../knowledge_resource/dbpedia_abstract_corpus  --max_seq_length 128   --entity_emb_size 100  --per_gpu_train_batch_size 16  --gradient_accumulation_steps 1  --learning_rate 3e-5  --weight_decay 0.01  --num_train_epochs 10  --max_steps 1000000  --warmup_steps -1  --max_save_checkpoints 20  --save_steps 30000  --logging_steps  1000  --seed  3407  --fp16  > pre-train.msl=128.emb_size=100.lr=3e-5.16 &


# 接着训练100万-300万共200万样本
nohup python -m torch.distributed.launch   --nproc_per_node=4    run_pretrain_kformers.py   --do_train  --model_type roberta  --model_name_or_path  roberta-large  --post_trained_checkpoint  ./output/checkpoint-150000/   --data_dir  ../knowledge_resource/dbpedia_abstract_corpus  --max_seq_length 128   --entity_emb_size 100  --per_gpu_train_batch_size 16  --gradient_accumulation_steps 1  --learning_rate 3e-5  --weight_decay 0.01  --num_train_epochs 10  --max_steps 1000000  --warmup_steps -1  --max_save_checkpoints 20  --save_steps 30000  --logging_steps  1000  --seed  3407  --fp16  --output_dir output_300w  > pre-train.msl=128.emb_size=100.lr=3e-5.16.300w &

# wikipedia
nohup python -m torch.distributed.launch   --nproc_per_node=4    run_pretrain_kformers.py   --do_train  --model_type roberta  --model_name_or_path  roberta-large  --data_dir  ../knowledge_resource/wikipedia   --max_seq_length 128   --entity_emb_size 100  --per_gpu_train_batch_size 40  --gradient_accumulation_steps 1  --learning_rate 3e-5  --weight_decay 0.01  --num_train_epochs 10  --max_steps 1000000  --warmup_steps -1  --max_save_checkpoints 20  --save_steps 30000  --logging_steps  1  --seed  3407  --fp16  > pre-train.wikipedia.msl=128.emb_size=100.lr=3e-5.40 &


 CUDA_VISIBLE_DEVICES='0' nohup  python run_pretrain_kformers.py   --do_train  --model_type roberta  --model_name_or_path  roberta-large  --data_dir  ../knowledge_resource/wikipedia   --max_seq_length 128   --entity_emb_size 100  --per_gpu_train_batch_size 32  --gradient_accumulation_steps 1  --learning_rate 3e-5  --weight_decay 0.01  --num_train_epochs 10  --max_steps 1000000  --warmup_steps -1  --max_save_checkpoints 20  --save_steps 30000  --logging_steps  1  --seed  3407  --fp16  > pre-train.wikipedia.msl=128.emb_size=100.lr=3e-5.32 &