pip install --user --editable .

python -m torch.distributed.launch --nproc_per_node=1  run_classifier_incr_inference.py  \
--do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  \
--eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 \
--gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   \
--num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin  \
 --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   \
--data_dir ../data/qualityControl/EntityPane_hh_offset   --output_dir output_clrv1  > log.CLRv1.128.1e-5 &


CUDA_VISIBLE_DEVICES="3" nohup  python run_classifier_incr_inference.py  \
--do_train --do_lower_case --seed 42 --max_seq_length 32 --train_batch_size 128  \
--eval_batch_size 512 --eval_per_epoch 3 --learning_rate 1e-5  --warmup_proportion 0.1 \
--gradient_accumulation_steps 1 --rel_pos_type 2 --weight_decay 0.0 --task_name ads   \
--num_train_epochs 3 --warmup_checkpoint ../checkpoints/CLRv1/model_100098.bin  \
 --vocab_file ../checkpoints/CLRv1/vocab.txt --bert_model  ../checkpoints/CLRv1   \
--data_dir ../data/qualityControl/EntityPane_hh_offset   --output_dir output_clrv1  --clip_data > log.KT-attn.CLRv1.128.1e-5 &