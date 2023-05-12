for len in 32
do
for lr in 1e-5 2e-5 3e-5 5e-5
do
	for alpha in 1.0 0.5 0.1
	do
        for beta in 0.001 0.01
        do
for seed in 12 42 57 1234 3407
do
CUDA_VISIBLE_DEVICES='0'  python run_KFormers.py  --task_name  sst2  --fp16 True   --model_type  KFormers  --backbone_model_type roberta  --knowledge_model_type  distilbert  --backbone_model_name_or_path  roberta-large  --knowledge_model_name_or_path  distilbert-base-uncased  --backbone_seq_length  128  --knowledge_seq_length $len  --qid_file  ../data/knowledge/pretrain/wikipedia_description/wikidata5m_des.wikipedia   --max_ent_num 4  --max_des_num 1  --train_batch_size 64   --valid_batch_size 256   --gradient_accumulation_steps 1   --max_steps -1   --update_K_module False   --data_dir ../data/gluedata/SST2/SST2_tagme  --output_dir S_15   --epochs 3   --learning_rate $lr  --eval_steps 50  --warmup_steps -1  --beta $beta   --seed $seed  > logs_sst/sst.len=$len.lr=$lr.alpha=$alpha.beta=$beta.seed=$seed
done
done
done
done
done
python occupation.py
