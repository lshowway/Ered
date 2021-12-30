CUDA_VISIBLE_DEVICES='0' nohup python main_GraphFormers.py --enable_gpu True --fp16 False --world_size 1 --eval_steps 3000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  2e-5   --warmup_lr  True   --warmup_step  2000 > log.CLRv1.916 &


CUDA_VISIBLE_DEVICES='0' nohup python run_GraphFormers.py  --eval_steps 500 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 2  > log.CLRv1.918.newcode &



nohup python -m torch.distributed.launch --nproc_per_node=4  run_GraphFormers.py  --eval_steps 100 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 1  > log.dataloader.batch=128*4*1 &


CUDA_VISIBLE_DEVICES='2' nohup python  run_GraphFormers.py  --eval_steps 2000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 1  --fp16 True  > log.922.fp16 &


nohup python -m torch.distributed.launch --nproc_per_node=2  run_GraphFormers.py  --eval_steps 100 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 1  --fp16 True  > log.distributed.fp16 &


CUDA_VISIBLE_DEVICES="0, 1" python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42 --max_seq_length 32 --origin_seq_length 16 --train_batch_size 128 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step 500

CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42 --max_seq_length 160 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.929.KT-attn.len=32-160.5e-5 &

CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.928.KT-attn.len=32-128.5e-5 &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 160 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 3e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.929.KT-attn.len=32-160.3e-5.distributed &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 160 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.929.KT-attn.len=32-160.5e-5.distributed &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.928.KT-attn.len=32-128.5e-5 &

######## 9.30
CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 3e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 10 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.930.KT-attn.len=32-128.3e-5 &



###### 10.04
CUDA_VISIBLE_DEVICES='2' nohup python  run_GraphFormers.py  --eval_steps 2000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 1  --fp16 True  > log.922.fp16 &


nohup python -m torch.distributed.launch --nproc_per_node=2  run_GraphFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 100   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.1004 &

nohup python -m torch.distributed.launch --nproc_per_node=4  run_GraphFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps 100 --epochs 3  --pretrain_lr 1e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.1004.32.64 &


######## 10.6
CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.10.06.KT-attn.len=32-128.5e-5.attn &

CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 10 --learning_rate 3e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.10.06.KT-attn.len=32-128.3e-5.KT-attn &


##### 10.8  KT-attn
CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.1008.KT-attn.len=32-128.5e-5 &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_classifier_incr_inference.py --do_train --fp16  --do_lower_case --seed 42 --max_seq_length 128 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 256 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.1008.KT-attn.len=32-128.5e-5 &

#### 10.8 KFormers
nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.64.2e-5 &

nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.32.1e-5 &

#### 10.9 KFormers
nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.32.5e-5 &


nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.64.5e-5 &


CUDA_VISIBLE_DEVICES='0'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.64.1e-5.128 &

CUDA_VISIBLE_DEVICES='1'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.knowledge.32.64.2e-5.128 &

nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.5e-5.128 &

nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.128.5e-5.128.k-layer-flase &

nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.5e-5.128 &

nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 500   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.128.5e-5.128 &

#### 10.10 KFormers
nohup python -m torch.distributed.launch --nproc_per_node=2  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.1e-5.128.920 &

nohup python -m torch.distributed.launch --nproc_per_node=2   run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.32.1e-5.128.915 &

nohup python -m torch.distributed.launch --nproc_per_node=2   run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.128.2e-5.128.930 &

#### 10.11 KFormers
nohup python -m torch.distributed.launch --nproc_per_node=2   run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.128.1e-5.128.1100 &

nohup python -m torch.distributed.launch --nproc_per_node=2   run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.128.5e-5.128.1130 &

# ==========
CUDA_VISIBLE_DEVICES='0'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.agg.32.32.1e-5 &

CUDA_VISIBLE_DEVICES='1'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.agg.32.32.2e-5 &

CUDA_VISIBLE_DEVICES='0'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 3e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.agg.32.32.3e-5 &

CUDA_VISIBLE_DEVICES='1'  nohup python   run_KFormers.py  --block_size 32  --k_seq_length 32  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.agg.32.32.5e-5 &

# ==========
CUDA_VISIBLE_DEVICES='0' nohup python  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.1e-5 &

CUDA_VISIBLE_DEVICES='1' nohup python  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.2e-5 &

CUDA_VISIBLE_DEVICES='0' nohup python run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 3e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.3e-5 &

CUDA_VISIBLE_DEVICES='1' nohup python  run_KFormers.py  --block_size 32  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.64.5e-5 &



#### 10.11 CLRv1
CUDA_VISIBLE_DEVICES='0'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.1e-5 &

CUDA_VISIBLE_DEVICES='1'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 2e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.2e-5 &

CUDA_VISIBLE_DEVICES='2'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 3e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.3e-5 &

CUDA_VISIBLE_DEVICES='3'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.5e-5 &

#### 10.12 KFormers
CUDA_VISIBLE_DEVICES='0, 1'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 256  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.256.1e-5 &

CUDA_VISIBLE_DEVICES='2, 3'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 256  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.256.2e-5 &

CUDA_VISIBLE_DEVICES='0, 1'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 256  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 3e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.256.3e-5 &

# ===
CUDA_VISIBLE_DEVICES='2, 3'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 512  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  > log.K-module.32.512.1e-5 &

# ==========
CUDA_VISIBLE_DEVICES='2'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 1e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  --seed 123 > log.K-module.32.128.1e-5.123 &

CUDA_VISIBLE_DEVICES='3'  nohup python  run_KFormers.py  --block_size 32  --k_seq_length 128  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 2e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  --seed 89  > log.K-module.32.128.2e-5.89 &


#### 10.12 GraphFormers
CUDA_VISIBLE_DEVICES='0' nohup python  run_GraphFormers.py  --eval_steps 1000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  1e-5  --gradient_accumulation_steps 1  --fp16 True  > log.GF.128.1e-5 &

CUDA_VISIBLE_DEVICES='1' nohup python  run_GraphFormers.py  --eval_steps 1000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  2e-5  --gradient_accumulation_steps 1  --fp16 True  > log.GF.128.2e-5 &

CUDA_VISIBLE_DEVICES='2' nohup python  run_GraphFormers.py  --eval_steps 1000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  3e-5  --gradient_accumulation_steps 1  --fp16 True  > log.GF.128.3e-5 &

CUDA_VISIBLE_DEVICES='3' nohup python  run_GraphFormers.py  --eval_steps 1000 --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr  5e-5  --gradient_accumulation_steps 1  --fp16 True  > log.GF.128.e-5 &

#### 10.12 CLRv1
CUDA_VISIBLE_DEVICES='0'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 123 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.1e-5.123 &

CUDA_VISIBLE_DEVICES='1'  nohup python    run_classifier_incr_inference.py  --do_train --do_lower_case --seed 89 --max_seq_length 32 --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir output_clrv1 > log.CLRv1.128.1e-5.89 &

#### 10.12 KT-attn
CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42  --origin_seq_length 16  --max_seq_length 64  --train_batch_size 32 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.KT-attn.16.64.64.5e-5 & (这个收敛)

CUDA_VISIBLE_DEVICES="2, 3" nohup  python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 32 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.KT-attn.32.64.64.5e-5 & (这个收敛吗？)

CUDA_VISIBLE_DEVICES="0, 1" nohup  python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 64 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.KT-attn.32.64.128.5e-5 & (这个收敛吗？)

CUDA_VISIBLE_DEVICES="2, 3" nohup  python run_classifier_incr_inference.py --do_train --fp16   --do_lower_case --seed 42 --max_seq_length 64 --origin_seq_length 32 --train_batch_size 64 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn_clrv1  --max_step -1 > log.KT-attn.32.64.128.5e-5 & 

#### 10.13 KT-attn
nohup python -m torch.distributed.launch --nproc_per_node=4 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 32 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --clip_data  > log.KT-attn.v3.no-mask.32.64.32*4.5e-5 & (v2)

CUDA_VISIBLE_DEVICES="0" nohup  python run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 128 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.64.128.5e-5.42 & 

CUDA_VISIBLE_DEVICES="1" nohup  python run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 123  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 128 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.64.128.5e-5.123 & 

CUDA_VISIBLE_DEVICES="2" nohup  python run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 89  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 128 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.64.128.5e-5.89 & 

nohup python -m torch.distributed.launch --nproc_per_node=2 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 128  --train_batch_size 64 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.128.128.5e-5.42 &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 123  --origin_seq_length 32  --max_seq_length 128  --train_batch_size 64 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.128.128.5e-5.123 &

nohup python -m torch.distributed.launch --nproc_per_node=2 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 89  --origin_seq_length 32  --max_seq_length 128  --train_batch_size 64 --eval_batch_size 800 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.128.128.5e-5.89 &


#### 10.24 bert-base-uncased QNLI
##### QNLI
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name QNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/QNLI/  --overwrite_output_dir > log.bert-base-uncased.QNLI.128.32.2e-5  & (这是bert-base-uncased)
##### QQP
CUDA_VISIBLE_DEVICES='0'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name QQP   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/QQP/  --overwrite_output_dir > log.bert-base-uncased.QQP.128.32.2e-5  &
##### MNLI
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name MNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/MNLI/  --overwrite_output_dir > log.bert-base-uncased.MNLI.128.32.2e-5  &
##### WNLI
CUDA_VISIBLE_DEVICES='2'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name WNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/WNLI/  --overwrite_output_dir > log.bert-base-uncased.WNLI.128.32.2e-5  &
##### RTE
CUDA_VISIBLE_DEVICES='3'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name RTE   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/RTE/  --overwrite_output_dir > log.bert-base-uncased.RTE.128.32.2e-5  &
##### SST2
CUDA_VISIBLE_DEVICES='0'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name sst2   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/SST2/  --overwrite_output_dir > log.bert-base-uncased.SST2.128.32.2e-5  &
##### CoLA
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name CoLA   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/CoLA/  --overwrite_output_dir > log.bert-base-uncased.CoLA.128.32.2e-5  &
##### MRPC
CUDA_VISIBLE_DEVICES='2'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name MRPC   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/MRPC/  --overwrite_output_dir > log.bert-base-uncased.MRPC.128.32.2e-5  &
##### STSB
CUDA_VISIBLE_DEVICES='3'  nohup  python run_glue.py --model_name_or_path  bert-base-uncased  --task_name stsb   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/WNLI/  --overwrite_output_dir > log.bert-base-uncased.STSB.128.32.2e-5  &


--load_best_model_at_end
#### 10.25 roberta-base
##### CoLA
CUDA_VISIBLE_DEVICES='0'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name CoLA   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 10.0  --output_dir ./glue_baseline_results/CoLA/  --overwrite_output_dir > log.roberta-base.CoLA.128.128.3e-5  &
##### STSB
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name stsb   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 32  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 10.0  --output_dir ./glue_baseline_results/WNLI/  --overwrite_output_dir > log.roberta-base.STSB.128.128.3e-5  &
##### QQP
CUDA_VISIBLE_DEVICES='2'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name QQP   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 5.0  --output_dir ./glue_baseline_results/QQP/  --overwrite_output_dir > log.roberta-base.QQP.128.128.3e-5  &
##### MNLI
CUDA_VISIBLE_DEVICES='3'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name MNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 5.0  --output_dir ./glue_baseline_results/MNLI/  --overwrite_output_dir > log.roberta-base.MNLI.128.128.3e-5  &
##### QNLI
CUDA_VISIBLE_DEVICES='0'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name QNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/QNLI/  --overwrite_output_dir > log.roberta-base.QNLI.128.128.5e-5  &
##### WNLI
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name WNLI   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 10.0  --output_dir ./glue_baseline_results/WNLI/  --overwrite_output_dir > log.roberta-base.WNLI.128.128.3e-5  &
##### RTE
CUDA_VISIBLE_DEVICES='2'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name RTE   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 10.0  --output_dir ./glue_baseline_results/RTE/  --overwrite_output_dir > log.roberta-base.RTE.128.128.3e-5  &
##### SST2
CUDA_VISIBLE_DEVICES='1'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name sst2   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 3.0  --output_dir ./glue_baseline_results/SST2/  --overwrite_output_dir > log.roberta-base.SST2.128.128.3e-5  &
##### MRPC
CUDA_VISIBLE_DEVICES='3'  nohup  python run_glue.py --model_name_or_path  roberta-base  --task_name MRPC   --do_train  --do_eval  --fp16 --max_steps -1  --logging_steps 100  --save_steps -1  --max_seq_length 128  --per_device_train_batch_size 128  --per_device_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 10.0  --output_dir ./glue_baseline_results/MRPC/  --overwrite_output_dir > log.roberta-base.MRPC.128.128.3e-5  &


#### 10.25 KFormers 
##### QQP
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py  --block_size 128  --k_seq_length 64  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 5  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  --task_name qqp --add_knowledge True  --data_dir ../data/gluedata/QQP/QQP_description_mention/  --output_dir ./glue_results/qqp  > log.KFormers.qqp.128.64.128.5e-5 & 
##### QNLI
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py  --block_size 128  --k_seq_length 16  --train_batch_size 128 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  --task_name qnli --data_dir ../data/gluedata/QNLIv2/QNLI_description_mention/  --output_dir ./glue_results/qnli  --add_knowledge True   > log.KFormers.qnli.128.16.128.5e-5 &

CUDA_VISIBLE_DEVICES='1, 2'  nohup python run_KFormers.py  --block_size 128  --k_seq_length 256  --train_batch_size 64 --valid_batch_size 512 --max_steps -1 --epochs 3  --pretrain_lr 5e-5  --eval_steps 1000   --gradient_accumulation_steps 1  --fp16 True  --task_name qnli --data_dir ../data/gluedata/QNLIv2/QNLI_description_mention/  --output_dir ./glue_results/qnli  --add_knowledge True   > log.KFormers.qnli.128.256.128.5e-5 &

##### WNLI

#### 10.13 KT-attn
nohup python -m torch.distributed.launch --nproc_per_node=4 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 64  --train_batch_size 32 --eval_batch_size 512 --eval_per_epoch 10 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --clip_data  > log.KT-attn.v3.no-mask.32.64.32*4.5e-5 & (v2)

CUDA_VISIBLE_DEVICES="0, 1, 2, 3" nohup  python run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 128  --max_seq_length 512  --train_batch_size 32 --eval_batch_size 128 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name sst2 --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/description_entityBias --output_dir output_qc_kt-attn  --max_step -1  --KT_attn  > log.KT-attn.v2.32.64.128.5e-5.42 &    

#### 11.1 KT-unilm，entry （别忘了加--fp16）
CUDA_VISIBLE_DEVICES='0' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/QC_description_mention   --output_dir output_kt_entry  >  log.KT-unilm-entry.128.1e-5.42  &


#### 11.1 KT-unilm，description, q+k=32,max_seq_length是q+k+description的长度
CUDA_VISIBLE_DEVICES='2' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 64  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/QC_description_mention   --output_dir output_kt_des  >  log.KT-unilm-des.128.1e-5.42  &

#### 11.3 KT-unilm，entry， EntityPane（别忘了加--fp16）
CUDA_VISIBLE_DEVICES='0' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane   --output_dir _1181154  >  log.KT-unilm-entry.pane.128.1e-5.42  &

CUDA_VISIBLE_DEVICES='1' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 123 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane   --output_dir t3  >  log.KT-unilm-entry.pane.128.1e-5.123  &

#### 11.3 KT-unilm，description, q+k=32,max_seq_length是q+k+description的长度, entityPane
CUDA_VISIBLE_DEVICES='2' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane   --output_dir tt1  >  log.KT-unilm-des.pane.32.1e-5.42  &

CUDA_VISIBLE_DEVICES='3' nohup python  run_KT_unilm.py  --do_train --fp16 --do_lower_case --seed 123 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane   --output_dir tt2  >  log.KT-unilm-des.pane.128.1e-5.123  &

#### 11.3 KT-attn, entityPane, description
nohup python -m torch.distributed.launch --nproc_per_node=2 run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 128  --train_batch_size 64 --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/EntityPane --output_dir 4_57_PM  --max_step -1  --KT_attn > log.KT-attn.pane.32.128.5e-5.42 &

CUDA_VISIBLE_DEVICES="2" nohup  python run_KT_attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 128  --train_batch_size 128 --eval_batch_size 128 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name quality_control --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1 --data_dir ../data/qualityControl/EntityPane --output_dir 4_35_PM  --max_step -1  --KT_attn  > log.KT-attn.pane.32.1.128.5e-5.42 & 

#### 11.8 PTM finetune code data+model+train args
--max_seq_length 32  --train_file ../data/qualityControl/CLRv1_format.train.tsv    --validation_file ../data/qualityControl/CLRv1_format.dev.tsv         --model_name_or_path  roberta-base  --output_dir ./output/qc/   --overwrite_output_dir   --do_train    --do_eval  --per_device_train_batch_size 32  --per_device_eval_batch_size  256  --learning_rate 2e-5  --num_train_epochs 3.0  --max_steps  -1  --logging_steps  1  --eval_steps  1

                 


evaluation_strategy

##### 11.8 CLRv1-our-entityPane+entry as input
CUDA_VISIBLE_DEVICES='3' nohup python  run_classifier_incr_inference.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 6 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane/E1E2   --output_dir PM_537  >  log.entityPane.128.1e-5.42.v3.e1e2  &

#### 11.8 our implement clrv1 finetune, same to KT-x
CUDA_VISIBLE_DEVICES='3' nohup python  run_clrv1_finetune.py  --do_train --fp16 --do_lower_case --seed 123 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 6  --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl   --output_dir PM_408  >  log.our-clrv1.128.1e-5.123  &

#### 11.8 bert-base EEM
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/raw   --output_dir=PM_931  > log.bert.ads  &

CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane   --output_dir=PM_841  > log.bert.ads.entityPane &

CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane/E1E2   --output_dir=PM_851  > log.bert.ads.entityPane.e1e2 &

#### 11.9 bert-base， EEM

##### q, k+e1+e2+k1+k2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_entry_key_e1e2   --output_dir=AM_116  > log.bert.ads.epek.-e1e2 &

#### 11.9 bert-base， EEM, description 
#### qk, des, d1-d2
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_des   --output_dir=PM_545  > log.bert.ads.des.d1-d2.32 &
#### qk, des, _d1d2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_des_d1d2   --output_dir=PM_546  > log.bert.ads.des.-d1d2.32 &
#### qk, des, _d1d2, EntityPane_hh_entry_des_e1e2d1d2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length=32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate=1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_entry_des_e1e2d1d2   --output_dir=AM_1136  > log.bert.ads.en-des.-e1e2d1d2.32 &

CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length  128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 1e-5  --num_train_epochs 3   --max_steps -1   --data_dir  ../data/qualityControl/EntityPane_des_d1d2   --output_dir PM_546  > log.bert.ads.des.-d1d2.128 &


#### 11.10 KT-attn,bert， EEM
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --origin_seq_length 32  --max_seq_length 64   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir PM_640 > log.kt-attn-bert.5e-5.64 &

CUDA_VISIBLE_DEVICES='2' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --origin_seq_length 32  --max_seq_length 128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir PM_640 > log.kt-attn-bert.5e-5.128 &


#### 11.11, bert, (qk, +entity+des, KT-attn, KFormers)

##### q, k
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name  quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length  32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3   --max_steps  -1   --data_dir  ../data/qualityControl/raw   --output_dir  PM_927  > log.bert.qc.qk.2e-5  &

#### qk, entry, des
CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length  64   --per_gpu_train_batch_size  128   --per_gpu_eval_batch_size 512  --learning_rate  3e-5  --num_train_epochs 3   --max_steps  -1   --data_dir  ../data/qualityControl/EntityPane_hh_entry_des_e1e2d1d2   --output_dir PM_934  > log.bert.ED.64.3e-5 &


##### q+e1/k1, k+e1/k2
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_entry   --output_dir=t1  > log.bert.ads.E.e1-e2.1e-5 &

##### qk, des
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 1e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_des   --output_dir=PM_343  > log.bert.ads.des.d1-d2.1e-5 &

CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 2e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_des   --output_dir=PM_343  > log.bert.ads.des.d1-d2.2e-5 &

CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_des   --output_dir=PM_345  > log.bert.ads.des.d1-d2.5e-5 &

#### KT-attn-unilm
CUDA_VISIBLE_DEVICES="3" nohup  python run_finetune_KT-attn.py  --do_train  --fp16  --do_lower_case --seed 42  --origin_seq_length 32  --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads  --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin --vocab_file ../checkpoints/CLRv1/vocab.txt   --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/EntityPane_hh_offset   --output_dir AM_118   > log.KT-attn.unilm.1e-5 &  


##### 11.22 CLRv1-our-v3 entityPane+entry as input
CUDA_VISIBLE_DEVICES='1' nohup python  run_classifier_incr_inference.py  --do_train --fp16 --do_lower_case --seed 42 --max_seq_length 32  --train_batch_size 128  --eval_batch_size 512 --eval_per_epoch 3 --learning_rate 5e-5  --warmup_proportion 0.1 --gradient_accumulation_steps 1 --rel_pos_type 2  --weight_decay 0.0 --task_name ads   --num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin   --vocab_file ../checkpoints/CLRv1/vocab.txt  --bert_model  ../checkpoints/CLRv1   --data_dir ../data/qualityControl/raw   --output_dir PM_457  >  log.clrv1.entityPane.5e-5  &


#### KT-attn-bert, EEM, SST2
##### sst2
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  sst2   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --origin_seq_length 128  --max_seq_length  128  --per_gpu_train_batch_size  128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs  3  --max_steps -1   --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_entry_one  --output_dir AM_15 > log.kt-attn-bert.sst2.5e-5 & 


#### 11.22 KG-emb-type, bert, 
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KG.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name  quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length  32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps  -1   --data_dir  ../data/qualityControl/EntityPane_type  --KG_file ../data/CKGv1.3/name_vec.vec  --TYPE_file  ../data/qualityControl/EntityPane_type/Type.set  --output_dir  PM_837  > log.KG-emb-type.5e-5  &


#### 12.06 LUKE, relation classification
python -m examples.cli    --model-file  ../checkpoints/luke_base_500k.tar.gz     --output-dir  output_2    relation-classification run    --data-dir  ../data/tacred     --train-batch-size  4     --gradient-accumulation-steps  8     --learning-rate  1e-5     --num-train-epochs  5     --fp16

python -m examples.cli     --model-file  ../checkpoints/luke_base_500k.tar.gz     --output-dir  output     entity-typing run     --data-dir  ../data/OpenEntity     --train-batch-size  2     --gradient-accumulation-steps  2     --learning-rate   1e-5     --num-train-epochs  3     --fp16


#### 12.7 roberta-base， SST2
CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name  sst2   --model_type roberta   --model_name_or_path roberta-base   --max_seq_length  128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs 3   --logging_steps  100  --max_steps  -1   --data_dir  ../data/gluedata/SST2  --output_dir  PM_234  > log.roberta.sst2.2e-5  &
##### LUKE-base-500k
CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name  sst2   --model_type roberta   --model_name_or_path ../checkpoints/luke_base_500k     --max_seq_length  128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 3   --logging_steps  100  --max_steps  -1   --data_dir  ../data/gluedata/SST2  --output_dir  PM_234  > log.LUKE.sst2.3e-5  &
##### roberta-base, EEM
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type roberta   --model_name_or_path roberta-base   --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 3e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/raw   --output_dir=t1  > log.roberta.EEM.3e-5 &
##### LUKE-base-500k, EEM
CUDA_VISIBLE_DEVICES='2' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type roberta   --model_name_or_path  ../checkpoints/luke_base_500k   --max_seq_length 32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 3e-5  --num_train_epochs 3   --max_steps -1   --data_dir=../data/qualityControl/raw   --output_dir t2  > log.LUKE.EEM.3e-5 &


#### 11.22 KG-emb-random, bert, sst
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KG.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name  sst2   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length  128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps  -1   --data_dir  ../data/gluedata/SST2/SST2_tagme/SST2_offset  --KG_file ../data/CKGv1.3/name_vec.vec  --output_dir  PM_837  > log.KG-emb-random.sst2.5e-5  &

#### 12.8 K-Adapter, Entity type
CUDA_VISIBLE_DEVICES='0' nohup python   run_finetune_openentity_adapter.py 
--model_type roberta  --model_name_or_path roberta-large   --config_name roberta-large  --do_train  --do_eval  --task_name=entity_type        --data_dir=../data/OpenEntity     --output_dir=./proc_data     --comment 'combine-adapter-trf'  --max_seq_length 256    --per_gpu_eval_batch_size 4  --per_gpu_train_batch_size 4   --learning_rate 1e-4  --gradient_accumulation_steps 1  --max_steps 12000  --model_name=roberta-large  --overwrite_output_dir   --overwrite_cache  --warmup_steps 120  --save_steps 1000  --freeze_bert=""   --freeze_adapter="True"   --adapter_size 768  --adapter_list "0,11,22"    --adapter_skip_layers 0    --meta_fac_adaptermodel=../checkpoints/fac-adapter/pytorch_model.bin --meta_lin_adaptermodel=../checkpoints/lin-adapter/pytorch_model.bin


#### 12.7 KFormers
##### EEM
CUDA_VISIBLE_DEVICES='3'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 500    --task_name  quality_control  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.EEM.5e-5.32.32 &  # add_knowledge
CUDA_VISIBLE_DEVICES='2'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 500    --task_name  quality_control  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.EEM.5e-5.32.32.k & 
##### SST2
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 128  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 100    --task_name  sst2  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.SST2.5e-5.128.128.k &
#### entity type， bert-base-uncased
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 32  --train_batch_size 32   --valid_batch_size 32   --max_steps 12000  --epochs 3  --learning_rate 1e-5  --eval_steps 500    --task_name  entity_type  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_1037   >  log.KFormers.bert.distilbert.OpenEntity.5e-5.256.128 &
#### entity type， OpenEntity, roberta-large,无knowledge，应该达到roberta-large的结果77.6， 75.0， 76.2
CUDA_VISIBLE_DEVICES='1'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --train_batch_size 4   --valid_batch_size 128   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1     --task_name  entity_type  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_103   >  log.KFormers.roberta-large.distilbert.OpenEntity.LUKE &
# add knowledge
CUDA_VISIBLE_DEVICES='3'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 2e-5  --eval_steps -1  --warmup_steps -1     --task_name  entity_type  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_103   >  log.KFormers.roberta-large.distilbert.OpenEntity.2e-5.256.128.32.K &

CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 3e-5  --eval_steps -1  --warmup_steps -1     --task_name  entity_type  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_103   >  log.KFormers.roberta-large.distilbert.OpenEntity.3e-5.256.128.32.K &

CUDA_VISIBLE_DEVICES='1'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 5e-5  --eval_steps -1  --warmup_steps -1     --task_name  entity_type  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_103   >  log.KFormers.roberta-large.distilbert.OpenEntity.5e-5.256.128.32.K &
#### entity typing, FIGER, no knowledge，train：2M，batch拉满
nohup python -m torch.distributed.launch --nproc_per_node=4  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 2e-5  --eval_steps -1  --warmup_steps -1     --task_name  figer  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/knowledge/FIGER  --output_dir AM_101   >  log.KFormers.roberta-large.distilbert.FIGER.2e-5.256.128.32 &

#### entity typing, FIGER, knowledge



#### 12,15  bert/roberta-base, EEM, entity or random entity
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 3e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/raw   --output_dir t0  > log.bert.EEM.3e-5.raw &

CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_entry   --output_dir tt  > log.bert.EEM.5e-5 &

CUDA_VISIBLE_DEVICES='3' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name=quality_control   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --max_seq_length 32   --per_gpu_train_batch_size=128   --per_gpu_eval_batch_size=512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps=-1   --data_dir=../data/qualityControl/EntityPane_hh_entry_random   --output_dir tt  > log.bert.EEM.5e-5.random &

#### 12.21 KFormers (KFormers_roberta_bert_knowledge)
##### EEM
CUDA_VISIBLE_DEVICES='3'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 500    --task_name  quality_control  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.EEM.5e-5.32.32 &  # add_knowledge
CUDA_VISIBLE_DEVICES='2'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 500    --task_name  quality_control  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/qualityControl/EntityPane_hh_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.EEM.5e-5.32.32.k & 
##### SST2
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 128  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 100    --task_name  sst2  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_offset  --output_dir AM_1037   >  log.KFormers.bert.distilbert.SST2.5e-5.128.128.k &
#### entity type， OpenEntity, bert-base-uncased
CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type bert  --knowledge_model_type  distilbert  --backbone_model_name_or_path  bert-base-uncased  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 32  --train_batch_size 32   --valid_batch_size 32   --max_steps 12000  --epochs 3  --learning_rate 1e-5  --eval_steps 500    --task_name  open_entity  --fp16 True  --add_knowledge False  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_1037   >  log.KFormers.bert.distilbert.OpenEntity.5e-5.256.128 &
#### entity type， OpenEntity, roberta-large
CUDA_VISIBLE_DEVICES='0, 1'  nohup python -m torch.distributed.launch --nproc_per_node=2 run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 512  --max_num_entity  1 --train_batch_size 16   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 2e-5  --eval_steps -1  --warmup_steps -1     --task_name  open_entity  --fp16 True  --add_knowledge True  --use_entity True   --data_dir ../data/knowledge/OpenEntity  --output_dir AM_113   >  log.KFormers.roberta-large.distilbert.OpenEntity.2e-5.256.512.32.K=1 &  # one GPU will OOM

CUDA_VISIBLE_DEVICES='0'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 2e-5  --eval_steps -1  --warmup_steps -1     --task_name  open_entity  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   >  log.KFormers.roberta-large.distilbert.OpenEntity.2e-5.256.64.32.K=1  &

CUDA_VISIBLE_DEVICES='4'  nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 3e-5  --eval_steps -1  --warmup_steps -1     --task_name  open_entity  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   >  log.KFormers.roberta-large.distilbert.OpenEntity.3e-5.256.--.32.K=1.F  &


#### entity typing, FIGER(Kadapter: 5e-6, 2048, 256)
CUDA_VISIBLE_DEVICES='0'  nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 32  --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate 5e-6  --eval_steps 500  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge False  --data_dir ../data/knowledge/FIGER_2W  --output_dir AM_20   >  log.KFormers.roberta-large.distilbert.FIGER.5e-6.256.--.32 &

CUDA_VISIBLE_DEVICES='1'  nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1 --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate  2e-5  --eval_steps -1  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge True  --data_dir ../data/knowledge/FIGER_2W  --output_dir AM_20   >  log.KFormers.roberta-large.distilbert.FIGER.2e-5.256.--.32 &

CUDA_VISIBLE_DEVICES='2'  nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 32  --max_num_entity  1 --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate  2e-5  --eval_steps -1  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge True  --data_dir ../data/knowledge/FIGER_2W  --output_dir AM_20   >  log.KFormers.roberta-large.distilbert.FIGER.2e-5.256.32.32 &

CUDA_VISIBLE_DEVICES='3'  nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 128  --max_num_entity  1 --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate  2e-5  --eval_steps -1  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge True  --data_dir ../data/knowledge/FIGER_2W  --output_dir AM_20   >  log.KFormers.roberta-large.distilbert.FIGER.2e-5.256.128.32 &


