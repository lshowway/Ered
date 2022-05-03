#### 3.1 entity type => OpenEntity
python run_KFormers.py    --model_type  KFormers  --backbone_model_type luke  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 4   --valid_batch_size 128   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --warmup_steps 0.06   --task_name  openentity  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13  --post_trained_checkpoint  /home/LAB/zhaoqh/phd4hh/KFormers/phase1_finetune_KFormers/post-train-output/checkpoint-15000


python -m cli  --model-file=../checkpoints/luke_large_500k.tar.gz  --output-dir=./output  entity-typing run   --data-dir ../data/OpenEntity   --train-batch-size 2 --gradient-accumulation-steps  2  --learning-rate  1e-5    --num-train-epochs 3  --fp16 --seed 12


# LUKE, openentity
python -m examples.cli   --model-file=../checkpoints/luke_large_500k.tar.gz --output-dir=./output entity-typing run --data-dir=../data/OpenEntity --train-batch-size=2 --gradient-accumulation-steps=2 --learning-rate=1e-5 --num-train-epochs=3



python run_luke.py  --data_dir  ../data/knowledge/FIGER   --output_dir  pm_20  --do_train  --baseline_model_name  roberta-large  --checkpoint_file  ../checkpoints/luke_large_500k  --train_batch_size 2  --eval_batch_size 64  --gradient_accumulation_steps  2   --learning_rate  1e-5  --num_train_epochs 3  --seed 12 

# TacRED
python -m examples.cli    --model-file=checkpoints/luke_large_500k.tar.gz     --output-dir=./output    relation-classification run     --data-dir=data/tacred    --train-batch-size=4     --gradient-accumulation-steps=8     --learning-rate=1e-5     --num-train-epochs=5  --checkpoit-file checkpoints/luke_large_500k.tar.gz
   

python run_luke_tacred.py  --data_dir  ../data/knowledge/tacred   --output_dir  pm_20  --do_train  --baseline_model_name  roberta-large  --checkpoint_file  ../checkpoints/luke_large_500k  --train_batch_size 4  --eval_batch_size 64  --gradient_accumulation_steps  8   --learning_rate  1e-5  --num_train_epochs 8  --seed 12   