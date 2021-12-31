# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# roberta-large, K-adapter, finetune on quality control data

CUDA_VISIBLE_DEVICES="0, 1" nohup python run_finetune_qualityControl_adapter_v1.py \






    --comment 'combine-adapter-trf' \
      \
     \
      \
     \
    --gradient_accumulation_steps=1 \
     \
     \
    --model_name=roberta-large  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=68 \
    --save_steps=10000 \

    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --fusion_mode="add" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="../ads_pretrain_output/ads_pretrain_maxlen-32_batch-128_lr-2e-05_warmup-2000_epoch-10_ads-fac-adapter/checkpoint-epoch-2-20154/pytorch_model.bin" > 0823.roberta.20154.adapter &


