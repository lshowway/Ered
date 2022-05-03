### 1. baseline: bert-base-uncased

#### 1.1 SST2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name sst2   --model_type bert   --model_name_or_path bert-base-uncased   --max_seq_length 128   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --logging_steps  500  --data_dir=../data/gluedata/SST2   --output_dir PM_91  > log.bert-base-uncased.SST2.128.128.5e-5  &
#### 1.2 EEM
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune.py  --do_train  --do_eval  --fp16  --evaluate_during_training  --overwrite_cache --task_name eem   --model_type bert   --model_name_or_path bert-base-uncased   --max_seq_length 32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 1e-5  --num_train_epochs 3   --max_steps -1  --logging_steps  500  --data_dir=../data/EEM/raw   --output_dir PM_931  > log.bert-base-uncased.EEM.32.128.1e-5  &

### 2. baseline (KFormers+noKnowledge): roberta-base

#### 2.1 SST2
nohup python run_KFormers.py  --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-base  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 64   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --task_name  sst2  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/gluedata/SST2/SST2_tagme  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.SST2.64.5e-5.128.noKnowledge &
#### 2.2 EEM
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-base  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --max_num_entity  1  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 1000    --task_name  eem  --fp16 True  --add_knowledge False  --update_K_module False   --data_dir ../data/EEM/EntityPane_hh_offset  --output_dir AM_1   >  log.KFormers.roberta-base.distilbert.EEM.128.5e-5.32.noKnowledge & 


### 3. baseline (KFormers+noKnowledge): roberta-large

#### 3.1 entity type => OpenEntity
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1     --task_name  openentity  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   >  logs/log.KFormers.roberta-large.distilbert.OpenEntity.32.1e-5.256.noKnowledge  &
##### 3.2 entity type => FIGER
nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256  --knowledge_seq_length 32  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps 1000  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge False  --data_dir ../data/knowledge/FIGER  --output_dir AM_20   >  logs/log.KFormers.roberta-large.distilbert.FIGER.32.256.1e-5.noKnowledge &
##### 3.3 relation classification => FewRel
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1     --task_name  fewrel  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/fewrel  --output_dir FR_14   >  logs/log.KFormers.roberta-large.distilbert.FewRel.32.1e-5.256.noKnowledge.10  &
##### 3.4 relation classification => TACRED
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1   --logging_steps 500  --task_name  tacred   --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/knowledge/tacred  --output_dir TR_13   >  logs/log.KFormers.roberta-large.distilbert.TACRED.32.1e-5.256.noKnowledge.3  &
##### 3.5 single sentence => SST2
nohup python run_KFormers.py  --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 64   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --task_name  sst2  --fp16 True  --add_knowledge False  --update_K_module False  --data_dir ../data/gluedata/SST2/SST2_tagme  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.SST2.64.5e-5.128.noKnowledge &
##### 3.6 sentence pair => EEM
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 32 --knowledge_seq_length 32  --max_num_entity  1  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 5e-5  --eval_steps 1000    --task_name  eem  --fp16 True  --add_knowledge False  --update_K_module False   --data_dir ../data/EEM/EntityPane_hh_offset  --output_dir AM_17   > noKnowledge

### a. baseline: KT, bert-base-uncased
only data needs to be changed
### b. baseline: KT, roberta-base
### c. baseline: KT, roberta-large

### i. baseline: KT-attn, bert-base-uncased
##### i.1 SST2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  sst2   --model_type bert   --model_name_or_path bert-base-uncased   --config_name bert-base-uncased  --origin_seq_length 128  --max_seq_length  128  --per_gpu_train_batch_size  128   --per_gpu_eval_batch_size 512  --learning_rate 2e-5  --num_train_epochs  3  --max_steps -1   --logging_steps  500  --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_offset  --output_dir AM_15 > log.kt-attn-bert.sst2.2e-5 & 
##### i.2 EEM
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  eem   --model_type bert   --model_name_or_path bert-base-uncased   --origin_seq_length 32  --max_seq_length 32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --logging_steps  500  --data_dir ../data/EEM/EntityPane_hh_offset  --output_dir PM_640 > log.kt-attn-bert.eem.5e-5.32.32 &

### i. baseline: KT-attn, roberta-base
#### i.1 SST2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  sst2   --model_type roberta   --model_name_or_path roberta-base    --origin_seq_length 128  --max_seq_length  128  --per_gpu_train_batch_size  128   --per_gpu_eval_batch_size 512  --learning_rate 1e-5  --num_train_epochs  3  --max_steps -1   --logging_steps  500  --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_offset  --output_dir AM_15 > log.kt-attn-roberta-base.sst2.1e-5 & 
##### i.2 EEM
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  eem   --model_type roberta   --model_name_or_path roberta-base   --origin_seq_length 32  --max_seq_length 32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --logging_steps  500  --data_dir ../data/EEM/EntityPane_hh_offset  --output_dir PM_640 > log.kt-attn-roberta-base.eem.5e-5.32.32 &

### i. baseline: KT-attn, roberta-large
#### i.1 SST2
CUDA_VISIBLE_DEVICES='1' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  sst2   --model_type roberta   --model_name_or_path roberta-large    --origin_seq_length 128  --max_seq_length  128  --per_gpu_train_batch_size  128   --per_gpu_eval_batch_size 512  --learning_rate 1e-5  --num_train_epochs  3  --max_steps -1   --logging_steps  500  --data_dir ../data/gluedata/SST2/SST2_tagme/SST2_offset  --output_dir AM_15 > log.kt-attn-roberta-large.sst2.1e-5 & 
##### i.2 EEM
CUDA_VISIBLE_DEVICES='0' nohup python run_finetune_KT-attn.py --do_train  --do_eval  --fp16  --evaluate_during_training  --task_name  eem   --model_type roberta   --model_name_or_path roberta-large   --origin_seq_length 32  --max_seq_length 32   --per_gpu_train_batch_size 128   --per_gpu_eval_batch_size 512  --learning_rate 5e-5  --num_train_epochs 3   --max_steps -1   --logging_steps  500  --data_dir ../data/EEM/EntityPane_hh_offset  --output_dir PM_640 > log.kt-attn-roberta-large.eem.5e-5.32.32 &

### x: our method: KFormers, roberta-large

##### x.1 使用knowledge, OpenEntity
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1     --task_name  openentity  --fp16 True  --add_knowledge True  --update_K_module False  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   >  logs/log.KFormers.roberta-large.distilbert.OpenEntity.32.1e-5.256.64.KFormers  &
##### x.2 使用knowledge, FIGER
nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256  --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps 500  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/FIGER  --output_dir AM_20   >  logs/log.KFormers.roberta-large.distilbert.FIGER.32.256.64.1e-5.KFormers &
#### x.3 relation classification， TACRED
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1   --logging_steps 500  --task_name  tacred   --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/tacred  --output_dir TR_14   >  logs/log.KFormers.roberta-large.distilbert.TACRED.32.1e-5.256.KFormers.3  &
##### x.4 使用knowledge，FewRel
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 3e-5  --eval_steps -1  --warmup_steps -1     --task_name  fewrel   --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/fewrel  --output_dir FR_14   >  logs/log.KFormers.roberta-large.distilbert.FewRel.32.3e-5.256.KFormers.10  & 
##### x.5 使用knowledge，SST2
2 GPU: nohup python run_KFormers.py  --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 64   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --task_name  sst2  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/gluedata/SST2  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.SST2.64.1e-5.128.KFormers &
#### x.6 EEM
nohup python run_KFormers.py  --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 128 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --task_name  eem  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/EEM/EEM_EntityPane_json  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.EEM.64.1e-5.128.KFormers &


### y: our method: KFormers (post-trained), roberta-large

##### y.1 使用knowledge, OpenEntity
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --post_trained_checkpoint  ../phase2_pretrain_KFormers/output/checkpoint-30000/  --backbone_seq_length 256  --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1     --task_name  openentity  --fp16 True  --add_knowledge True  --update_K_module False  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   >  logs/log.KFormers.post-3W.distilbert.OpenEntity.32.1e-5.256.64.KFormers  &
##### y.2 使用knowledge, FIGER
nohup python  run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --post_trained_checkpoint  ../phase2_pretrain_KFormers/output/checkpoint-30000/   --backbone_seq_length 256  --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 256   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps 500  --warmup_steps -1  --gradient_accumulation_steps  1   --task_name  figer  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/FIGER  --output_dir AM_20   >  logs/log.KFormers.post-3W.distilbert.FIGER.32.256.64.1e-5.KFormers &
#### y.3 relation classification， TACRED
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --post_trained_checkpoint  ../phase2_pretrain_KFormers/output/checkpoint-30000/   --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --warmup_steps -1   --logging_steps 500  --task_name  tacred   --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/tacred  --output_dir TR_14   >  logs/log.KFormers.post-3W.distilbert.TACRED.32.1e-5.256.KFormers.3  &
##### y.4 使用knowledge，FewRel
nohup python run_KFormers.py    --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --post_trained_checkpoint  ../phase2_pretrain_KFormers/output/checkpoint-30000/  --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 32   --valid_batch_size 128   --max_steps -1  --epochs 10  --learning_rate 3e-5  --eval_steps -1  --warmup_steps -1     --task_name  fewrel   --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/fewrel  --output_dir FR_14   >  logs/log.KFormers.post-3W.distilbert.FewRel.32.3e-5.256.KFormers.10  & 


#### y.6 EEM
nohup python run_KFormers.py  --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --post_trained_checkpoint  ../phase2_pretrain_KFormers/output/checkpoint-30000/  --backbone_seq_length 128 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 128   --valid_batch_size 512   --max_steps -1  --epochs 3  --learning_rate 1e-5  --eval_steps -1  --task_name  eem  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/EEM/EEM_EntityPane_json  --output_dir AM_146   >  log.KFormers.roberta-large.distilbert.EEM.64.1e-5.128.KFormers &





















#### Kadapter (这个代码的fp16没有改好mdzz)
##### roberta-large, EEM
nohup python run_finetune_qualityControl_adapter_roberta.py --model_type roberta   --model_name_or_path roberta-large  --do_train  --do_eval --fp16 --task_name eem    --output_dir ./output   --max_seq_length 32  --per_gpu_eval_batch_size 512  --per_gpu_train_batch_size 128  --learning_rate 2e-5 --gradient_accumulation_steps 1     --max_steps -1    --num_train_epochs 3  --overwrite_output_dir  --overwrite_cache     --warmup_steps 317    --save_steps -1  --logging_steps  500    --freeze_adapter "True"    --adapter_size 768     --adapter_list "0,11,22"    --fusion_mode "add"    --adapter_skip_layers 0    --meta_fac_adaptermodel ../checkpoints/fac-adapter/pytorch_model.bin   --data_dir ../data/EEM/raw  > logs/kadapter.roberta-large.EEM.2e-5 &
##### roberta-large, SST2
CUDA_VISIBLE_DEVICES='0, 1' nohup python -m torch.distributed.launch  --nproc_per_node=2   run_finetune_qualityControl_adapter_roberta.py   --model_type roberta   --model_name_or_path roberta-large  --do_train  --do_eval   --fp16   --task_name sst2    --output_dir ./PM359   --max_seq_length 128  --per_gpu_eval_batch_size 256  --per_gpu_train_batch_size 64  --learning_rate 2e-5 --gradient_accumulation_steps 1     --max_steps -1    --num_train_epochs 3  --overwrite_output_dir  --overwrite_cache     --warmup_steps 47    --save_steps -1  --logging_steps  500   --freeze_adapter "True"    --adapter_size 768     --adapter_list "0,11,22"    --fusion_mode "add"    --adapter_skip_layers 0    --meta_fac_adaptermodel ../checkpoints/fac-adapter/pytorch_model.bin   --data_dir ../data/gluedata/SST2  >  logs/kadapter.roberta-large.SST2.2e-5 &





#### 2022.4.26
### LUKE+distilbert
python run_KFormers.py    --model_type  KFormers  --backbone_model_type luke  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased   --backbone_seq_length 256 --knowledge_seq_length 20  --max_ent_num  1  --max_des_num 1  --train_batch_size 4  --gradient_accumulation_steps 1 --valid_batch_size 128   --max_steps -1  --task_name  openentity  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   --epochs 3  --eval_steps -1  --learning_rate 1e-5  --warmup_steps 0.06  --alpha  1.0 --beta 1.0 --seed 42  (78.68--77.8)

python run_KFormers.py    --model_type  KFormers  --backbone_model_type luke  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased  --post_trained_checkpoint  /home/LAB/zhaoqh/phd4hh/KFormers/phase1_finetune_KFormers/PT_FIGER_update/checkpoint-44000  --backbone_seq_length 256 --knowledge_seq_length 64  --max_num_entity  1  --train_batch_size 4  --gradient_accumulation_steps 1 --valid_batch_size 128   --max_steps -1  --task_name  openentity  --fp16 True  --add_knowledge True  --update_K_module False  --data_dir ../data/knowledge/OpenEntity  --output_dir PM_13   --epochs 3  --eval_steps -1  --learning_rate 1e-5  --warmup_steps 0.06  --seed 12

#### LUKE+Adapter
python run_luke.py  --data_dir  ../data/knowledge/OpenEntity   --output_dir  pm_20  --do_train  --baseline_model_name  roberta-large  --checkpoint_file  ../checkpoints/luke_large_500k  --train_batch_size 4  --eval_batch_size 64  --gradient_accumulation_steps  1   --learning_rate  1e-5  --num_train_epochs 3  --fp16 True --seed 12 







