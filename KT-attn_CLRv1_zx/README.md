## Environment

The recommended way to run the code is using docker:
```bash
alias=`whoami | cut -d'.' -f2`; sudo docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel bash
```

The docker is initialized by:
```bash
apt-get update
apt-get install -y vim wget ssh

PWD_DIR=$(pwd)
cd $(mktemp -d)
git clone -q https://github.com/NVIDIA/apex.git
cd apex
git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
python setup.py install --user --cuda_ext --cpp_ext
cd $PWD_DIR

pip install --user scikit-learn numpy scipy tqdm regex pudb tensorboardX
```

Clone this repo and install requirements:
```bash
cd ~/code
git clone https://github.com/donglixp/unilm-alexander.git
cd ~/code/unilm-alexander/src-finetune
```


## Download checkpoint

```bash
wget -O /path/to/model/unilm2-base-uncased.bin https://unilm.blob.core.windows.net/ckpt/unilm2-base-uncased.bin
```


## Fine-tune for SQuAD 2.0 (span extraction, and no answer classification)

### Donwload data

```bash
# Set a path to save training/dev dataset.
export DATASET_PATH=/path/to/save/your/dataset
# training set
wget -O $DATASET_PATH/train-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
# dev set
wget -O $DATASET_PATH/dev-v2.0.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

### Run fine-tuning

```bash
# Set path to read training/dev dataset that you save in last step
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

python run_squad.py --bert_model bert-base-uncased --do_lower_case \
        --do_train --do_predict --fp16 \
        --train_file $DATASET_PATH/train-v2.0.json \
        --predict_file $DATASET_PATH/dev-v2.0.json \
        --train_batch_size 32 --learning_rate 3e-5 \
        --num_train_epochs 4.0 --gradient_accumulation_steps 2 \
        --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
        --version_2_with_negative --seed 0 --warmup_checkpoint $CHECKPOINT \
        --fast_qkv --rel_pos_type 2
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The fine-tuning takes about 1 hour on a v100-16GB card. The evaluation results are dumped to a json file. The `best_f1` field shows the F1 score on SQuAD 2.0.

```bash
cat $OUTPUT_PATH/best_performance.json

{
  "exact": 83.10452286700918,
  "f1": 86.01887483541677,
  "total": 11873,
  "HasAns_exact": 78.98110661268556,
  "HasAns_f1": 84.81816817154248,
  "HasAns_total": 5928,
  "NoAns_exact": 87.2161480235492,
  "NoAns_f1": 87.2161480235492,
  "NoAns_total": 5945,
  "best_exact": 83.23085993430473,
  "best_exact_thresh": -5.392619609832764,
  "best_f1": 86.01887483541616,
  "best_f1_thresh": -3.5195231437683105
}
```


## Fine-tune for SQuAD 1.1 (span extraction)

### Donwload data

```bash
# Set a path to save training/dev dataset.
export DATASET_PATH=/path/to/save/your/dataset
# training set
wget -O $DATASET_PATH/train-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
# dev set
wget -O $DATASET_PATH/dev-v1.1.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

### Run fine-tuning

```bash
# Set path to read training/dev dataset that you save in last step
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

python run_squad.py --bert_model bert-base-uncased --do_lower_case \
        --do_train --do_predict --fp16 \
        --train_file $DATASET_PATH/train-v1.1.json \
        --predict_file $DATASET_PATH/dev-v1.1.json \
        --train_batch_size 32 --learning_rate 2e-5 \
        --num_train_epochs 3.0 --gradient_accumulation_steps 2 \
        --max_seq_length 384 --doc_stride 128 --output_dir $OUTPUT_PATH \
        --seed 0 --warmup_checkpoint $CHECKPOINT \
        --fast_qkv --rel_pos_type 2
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The fine-tuning takes about 1 hour on a v100-16GB card. The evaluation results are dumped to a json file. The `best_f1` field shows the F1 score on SQuAD 1.1.

```bash
cat $OUTPUT_PATH/performance.json

{
  "exact": 86.97256385998108,
  "f1": 93.18527304372044,
  "total": 10570,
  "HasAns_exact": 86.97256385998108,
  "HasAns_f1": 93.18527304372044,
  "HasAns_total": 10570
}
```


## Fine-tune for MNLI (pairwise classification)

### Donwload data

```bash
# Set a path to save dataset.
export DATASET_PATH=/path/to/save/your/dataset
wget -O $DATASET_PATH/mnli.zip https://unilm.blob.core.windows.net/nlu-data/mnli.zip
cd $DATASET_PATH ; unzip mnli.zip
```

### Run fine-tuning

```bash
# Set path to read training/dev dataset that you save in last step
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

export BSZ=16
export LR=1e-5
export EPOCH=5
export WD=0.1
export WM=0.1
python run_classifier.py --task_name mnli --do_train --do_eval --do_lower_case --fp16 \
    --bert_model bert-base-uncased --data_dir $DATASET_PATH \
    --max_seq_length 128 --train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH \
    --output_dir $OUTPUT_PATH --seed -1 --warmup_checkpoint $CHECKPOINT \
    --eval_per_epoch 2 --eval_batch_size 32 --weight_decay $WD \
    --fast_qkv --rel_pos_type 2 --warmup_proportion $WM
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate 1e-5`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The fine-tuning takes about 2-3 hour on a v100-16GB card. The evaluation results are dumped to a text file. The `eval_accuracy` field shows the accuracy on MNLI dev set.

```bash
cat $OUTPUT_PATH/eval_results.txt

{"eval_loss": 0.4062523857390065, "eval_accuracy": 0.8480896586856852, "global_step": 12271}
{"eval_loss": 0.3554957883754072, "eval_accuracy": 0.8677534386143657, "global_step": 24542}
{"eval_loss": 0.34841402035194424, "eval_accuracy": 0.8732552215995925, "global_step": 36813}
{"eval_loss": 0.34021294311125816, "eval_accuracy": 0.8789607743250127, "global_step": 49084}
{"eval_loss": 0.35545473378333675, "eval_accuracy": 0.8847682119205298, "global_step": 61355}
{"eval_loss": 0.3517897416403705, "eval_accuracy": 0.8833418237391747, "global_step": 73626}
{"eval_loss": 0.38313961960982035, "eval_accuracy": 0.8867040244523688, "global_step": 85897}
{"eval_loss": 0.3999508786279138, "eval_accuracy": 0.8828323993886907, "global_step": 98168}
{"eval_loss": 0.4333626315337439, "eval_accuracy": 0.8855832908813042, "global_step": 110439}
{"eval_loss": 0.43924477046009774, "eval_accuracy": 0.8847682119205298, "global_step": 122710}
{"eval_loss": 0.4392478023367518, "eval_accuracy": 0.884666327050433, "global_step": 122715}
```


## Fine-tune for SST-2 (text classification)

### Donwload data

```bash
# Set a path to save dataset.
export DATASET_PATH=/path/to/save/your/dataset
wget -O $DATASET_PATH/sst-2.zip https://unilm.blob.core.windows.net/nlu-data/sst-2.zip
cd $DATASET_PATH ; unzip sst-2.zip
```

### Run fine-tuning

```bash
# Set path to read training/dev dataset that you save in last step
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

export BSZ=32
export LR=7e-6
export EPOCH=15
export WD=0.01
export WM=0.2
python run_classifier.py --task_name sst-2 --do_train --do_eval --do_lower_case --fp16 \
    --bert_model bert-base-uncased --data_dir $DATASET_PATH \
    --max_seq_length 128 --train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH \
    --output_dir $OUTPUT_PATH --seed -1 --warmup_checkpoint $CHECKPOINT \
    --eval_per_epoch 2 --eval_batch_size 32 --weight_decay $WD \
    --fast_qkv --rel_pos_type 2 --warmup_proportion $WM
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The evaluation results are dumped to a text file. The `eval_accuracy` field shows the accuracy on SST-2 dev set.

```bash
cat $OUTPUT_PATH/eval_results.txt

# ...
{"eval_loss": 0.21813086100987025, "eval_accuracy": 0.9495412844036697, "global_step": 15780}
{"eval_loss": 0.21746601377214705, "eval_accuracy": 0.9506880733944955, "global_step": 16832}
{"eval_loss": 0.24659361158098494, "eval_accuracy": 0.9461009174311926, "global_step": 17884}
{"eval_loss": 0.2205065999712263, "eval_accuracy": 0.9461009174311926, "global_step": 18936}
{"eval_loss": 0.2701935086931501, "eval_accuracy": 0.944954128440367, "global_step": 19988}
{"eval_loss": 0.2922210693359375, "eval_accuracy": 0.9415137614678899, "global_step": 21040}
{"eval_loss": 0.27772988591875347, "eval_accuracy": 0.9472477064220184, "global_step": 22092}
{"eval_loss": 0.23922027860369002, "eval_accuracy": 0.9506880733944955, "global_step": 23144}
{"eval_loss": 0.26636198588779997, "eval_accuracy": 0.9472477064220184, "global_step": 24196}
{"eval_loss": 0.27964137281690327, "eval_accuracy": 0.9426605504587156, "global_step": 25248}
{"eval_loss": 0.2986824171883719, "eval_accuracy": 0.9415137614678899, "global_step": 26300}
{"eval_loss": 0.2994298253740583, "eval_accuracy": 0.9392201834862385, "global_step": 27352}
{"eval_loss": 0.30513419423784527, "eval_accuracy": 0.9392201834862385, "global_step": 28404}
{"eval_loss": 0.2897782155445644, "eval_accuracy": 0.9426605504587156, "global_step": 29456}
{"eval_loss": 0.2885592835290091, "eval_accuracy": 0.9426605504587156, "global_step": 30508}
{"eval_loss": 0.299382141658238, "eval_accuracy": 0.9426605504587156, "global_step": 31560}
```

## Fine-tune for STS-B (regression with MSELoss)

### Donwload data

```bash
# Set a path to save dataset.
export DATASET_PATH=/path/to/save/your/dataset
wget -O $DATASET_PATH/sts-b.zip https://unilm.blob.core.windows.net/nlu-data/sts-b.zip
cd $DATASET_PATH ; unzip sts-b.zip
```

### Run fine-tuning

```bash
# Set path to read training/dev dataset that you save in last step
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

export BSZ=16
export LR=1e-5
export EPOCH=15
export WD=0.1
export WM=0.2
python run_classifier.py --task_name sts-b --do_train --do_eval --do_lower_case --fp16 \
    --bert_model bert-base-uncased --data_dir $DATASET_PATH \
    --max_seq_length 128 --train_batch_size $BSZ --learning_rate $LR --num_train_epochs $EPOCH \
    --output_dir $OUTPUT_PATH --seed -1 --warmup_checkpoint $CHECKPOINT \
    --eval_per_epoch 2 --eval_batch_size 32 --weight_decay $WD \
    --fast_qkv --rel_pos_type 2 --warmup_proportion $WM
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The evaluation results are dumped to a text file. The `eval_accuracy` field shows the accuracy on STS-B dev set.
```bash
cat $OUTPUT_PATH/eval_results.txt

# ...
{"eval_loss": 0.40039801756118204, "eval_accuracy": {"pearson": 0.9115469827552549, "spearmanr": 0.9085460664497674, "corr": 0.9100465246025111}, "global_step": 4475}
{"eval_loss": 0.4138317642376778, "eval_accuracy": {"pearson": 0.9090268761901612, "spearmanr": 0.9065655073185608, "corr": 0.907796191754361}, "global_step": 4654}
{"eval_loss": 0.4121048318895888, "eval_accuracy": {"pearson": 0.9092945885939001, "spearmanr": 0.9073122833291456, "corr": 0.9083034359615229}, "global_step": 4833}
{"eval_loss": 0.4307927606587714, "eval_accuracy": {"pearson": 0.9100446854312907, "spearmanr": 0.907658301631139, "corr": 0.9088514935312149}, "global_step": 5012}
{"eval_loss": 0.4211669833736217, "eval_accuracy": {"pearson": 0.9093186182143056, "spearmanr": 0.9073525581585308, "corr": 0.9083355881864181}, "global_step": 5191}
{"eval_loss": 0.4171488066302969, "eval_accuracy": {"pearson": 0.9094322827866506, "spearmanr": 0.9072777431843719, "corr": 0.9083550129855112}, "global_step": 5370}
{"eval_loss": 0.4171674911012041, "eval_accuracy": {"pearson": 0.9094278065157364, "spearmanr": 0.9072621655271024, "corr": 0.9083449860214194}, "global_step": 5385}
```
## Fine-tune for TQnA (pairwise classification)

### Data

The data folder is organized as in `~/code/unilm-alexander/src-finetune/tqna.json`:
```json
{
    "name": "tqna",
    "data": {
        "finetune_dataset": "train/uhrs-dataset-triplet.tsv",
        "evaluation_sets": {
            "bing-ann-eval": "eval/bing-ann-eval.tsv",
            "google-malta-1m-eval": "eval/google-malta-1m-eval.tsv",
            "uhrs-eval": "eval/uhrs-eval.tsv"
        }
    }
}
```

```bash
# Set path to read training/dev datasets
export DATASET_PATH=/path/to/read/your/dataset
# path of training set
$DATASET_PATH/train/uhrs-dataset-triplet.tsv
# path of three evaluation sets
$DATASET_PATH/eval/bing-ann-eval.tsv
$DATASET_PATH/eval/google-malta-1m-eval.tsv
$DATASET_PATH/eval/uhrs-eval.tsv
```

### Run fine-tuning

```bash
# Set path to read training/dev datasets
export DATASET_PATH=/path/to/read/your/dataset
# Set path to save the finetuned model and result score
export OUTPUT_PATH=/path/to/save/result_of_finetuning
# Set path to the model checkpoint you need to test 
export CHECKPOINT=/path/to/model/unilm2-base-uncased.bin

export PYTORCH_PRETRAINED_BERT_CACHE=/path/to/bert-cased-pretrained-cache
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node=1 run_qp.py --bert_model bert-base-uncased --do_lower_case \
        --name tqdn --data_dir $DATASET_PATH --config-file tqna.json \
        --do_train --do_eval --fp16 --do_finetune \
        --gradient_accumulation_steps 1 --max_seq_length 128 --output_dir $OUTPUT_PATH \
        --seed 0 --warmup_checkpoint $CHECKPOINT \
        --fast_qkv --rel_pos_type 2 --eval_batch_size 32 \
        --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 3 --warmup_proportion 0.1
```

- `CUDA_VISIBLE_DEVICES=0`: visible GPU ids
- `--learning_rate`: learning rate
- `--num_train_epochs`: the number of epochs for fine-tuning
- `--train_batch_size`: batch size
- `--seed 0`: random seed
- `--fast_qkv --rel_pos_type 2` : flags used for v2.1; need to be removed for other models

### Evaluation results

The fine-tuning takes about 1.5 hour/epoch on a v100-16GB card. The evaluation results are dumped to json files.

```bash
tail $OUTPUT_PATH/saved_models/*/auc_*.json

==> saved_models/tqdn/auc_1.json <==
{"bing-ann-eval": 0.7614380237834113, "google-malta-1m-eval": 0.7095428228708516, "uhrs-eval": 0.7945285953958545}
==> saved_models/tqdn/auc_2.json <==
{"bing-ann-eval": 0.775174495035698, "google-malta-1m-eval": 0.7198546781287485, "uhrs-eval": 0.7995868619148436}
==> saved_models/tqdn/auc_3.json <==
{"bing-ann-eval": 0.7835761962520531, "google-malta-1m-eval": 0.7188856257497, "uhrs-eval": 0.7980754119945634}
```
