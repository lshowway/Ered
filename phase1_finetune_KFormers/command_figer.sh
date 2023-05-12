for len in 32
do
	for ent_num in 4
	do
		for lr in 1e-5 5e-5 3e-5 2e-5
		do
			for seed in 12
			do
				for alpha in 1.0
				do
				for beta  in 0.001
				do
python  -m torch.distributed.launch --nproc_per_node=4  run_KFormers.py  --task_name figer   --fp16 True   --model_type KFormers --backbone_model_type luke   --knowledge_model_type distilbert --backbone_model_name_or_path roberta-large --knowledge_model_name_or_path distilbert-base-uncased  --backbone_seq_length 128  --knowledge_seq_length $len    --qid_file  ../data/knowledge/pretrain/wikipedia_description/wikidata5m_des.wikipedia   --max_ent_num $ent_num  --max_des_num 1 --train_batch_size 64  --gradient_accumulation_steps 8  --valid_batch_size 256 --max_steps -1    --update_K_module False  --data_dir ../data/knowledge/FIGER  --output_dir FIGER_output  --epochs 2 --eval_steps 50  --save_steps -1  --learning_rate $lr --warmup_steps -1  --alpha $alpha  --beta $beta  --seed $seed >  logs_luke_figer/figer.len=$len.ent_num=$ent_num.lr=$lr.64*4*16.alpha=$alpha.beta=$beta.seed=$seed
done
done
done
done
done
done
python occupation.py
