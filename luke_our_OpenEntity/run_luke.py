import numpy as np
import random
import sys
sys.path.append("..")
import json

import contextlib
import logging
import os

import torch
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AdamW, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup


from data_utils import load_examples
from parameters import parse_args
from transformers import RobertaTokenizer
from luke_modeling import LukeForEntityTyping
from model_utils import ModelArchive

from util import NullLogger, CometLogger


logger = logging.getLogger(__name__)





class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps, step_callback=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        self.step_callback = step_callback

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)

    def train(self):
        model = self.model
        optimizer = self.optimizer

        if self.args.fp16:
            from apex import amp

            model, optimizer = amp.initialize(
                model,
                optimizer,
                opt_level=self.args.fp16_opt_level,
                min_loss_scale=self.args.fp16_min_loss_scale,
                max_loss_scale=self.args.fp16_max_loss_scale,
            )  # model and optimizer fp16初始化

        if self.args.local_rank != -1:  # 分布式
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        epoch = 0
        global_step = 0
        tr_loss = 0.0

        num_workers = torch.cuda.device_count()

        def maybe_no_sync(step):
            if (
                hasattr(model, "no_sync")
                and num_workers > 1
                and (step + 1) % self.args.gradient_accumulation_steps != 0
            ):
                return model.no_sync()
            else:
                return contextlib.ExitStack()

        model.train()
        # 准备好data, optimizer, scheduler, model现在开始训练(这个disable是干嘛的，分布式的时候不使用tqdm？
        set_seed(self.args)
        with tqdm(total=self.num_train_steps, disable=self.args.local_rank not in (-1, 0)) as pbar:
            while True:
                for step, batch in enumerate(self.dataloader):
                    inputs = {k: v.to(self.args.device) for k, v in self._create_model_arguments(batch).items()}
                    outputs = model(**inputs)  # 训练计算
                    loss = outputs[0]
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    with maybe_no_sync(step):  # ？
                        if self.args.fp16:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()

                    tr_loss += loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        if self.args.max_grad_norm != 0.0:
                            if self.args.fp16:
                                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                        self.optimizer.step()
                        self.scheduler.step()
                        model.zero_grad()

                        pbar.set_description("epoch: %d loss: %.7f" % (epoch, loss.item()))
                        pbar.update()
                        global_step += 1

                        if self.step_callback is not None:
                            self.step_callback(model, global_step)

                        if (
                            self.args.local_rank in (-1, 0)
                            and self.args.output_dir
                            and self.args.save_steps > 0
                            and global_step % self.args.save_steps == 0
                        ):
                            output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

                            if hasattr(model, "module"):
                                torch.save(model.module.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                            else:
                                torch.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

                        if global_step == self.num_train_steps:
                            break

                if global_step == self.num_train_steps:
                    break
                epoch += 1

        logger.info("global_step = %s, average loss = %s", global_step, tr_loss / global_step)

        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,  # 不是no_decay，weight_decay=0.01
            },
            {
                "params": [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,  # no_decay里面的参数不需要正则
            },
        ]
        return AdamW(
            optimizer_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_eps,
            betas=(self.args.adam_b1, self.args.adam_b2),
            correct_bias=self.args.adam_correct_bias,
        )  # 返回一个优化器对象

    def _create_scheduler(self, optimizer):
        warmup_steps = int(self.num_train_steps * self.args.warmup_proportion)  # warmup_step： 0.06
        if self.args.lr_schedule == "warmup_linear":
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)
        if self.args.lr_schedule == "warmup_constant":
            return get_constant_schedule_with_warmup(optimizer, warmup_steps)

        raise RuntimeError("Unsupported scheduler: " + self.args.lr_schedule)

    def _create_model_arguments(self, batch):
        return batch




def evaluate(args, model, fold="dev", output_file=None):
    dataloader, _, _, label_list = load_examples(args, fold=fold)
    model.eval()

    all_logits = []
    all_labels = []
    for idx, batch in tqdm(enumerate(dataloader), desc=fold):
        inputs = {k: v.to(args.device) for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            logits = model(**inputs)

        logits = logits.detach().cpu().tolist()
        labels = batch["labels"].to("cpu").tolist()

        all_logits.extend(logits)
        all_labels.extend(labels)

    all_predicted_indexes = []
    all_label_indexes = []
    for logits, labels in zip(all_logits, all_labels):
        all_predicted_indexes.append([i for i, v in enumerate(logits) if v > 0])
        all_label_indexes.append([i for i, v in enumerate(labels) if v > 0])

    if output_file:
        with open(output_file, "w") as f:
            for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
                data = dict(
                    predictions=[label_list[ind] for ind in predicted_indexes],
                    labels=[label_list[ind] for ind in label_indexes],
                )
                f.write(json.dumps(data) + "\n")

    num_predicted_labels = 0
    num_gold_labels = 0
    num_correct_labels = 0

    for predicted_indexes, label_indexes in zip(all_predicted_indexes, all_label_indexes):
        num_predicted_labels += len(predicted_indexes)
        num_gold_labels += len(label_indexes)
        num_correct_labels += len(frozenset(predicted_indexes).intersection(frozenset(label_indexes)))

    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.0

    recall = num_correct_labels / num_gold_labels
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return dict(precision=precision, recall=recall, f1=f1)




def do_train(args, train_dataloader, model):
    # train
    results = {}
    if True:  # args.do_train:
        num_train_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps  # 50
        num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)  # 1000

        best_dev_f1 = [-1]
        best_weights = [None]

        def step_callback(model, global_step):
            if global_step % num_train_steps_per_epoch == 0 and args.local_rank in (0, -1):
                # if global_step > 0:
                epoch = int(global_step / num_train_steps_per_epoch - 1)
                dev_results = evaluate(args, model, fold="dev")
                test_results = evaluate(args, model, fold="test")
                args.experiment.log_metrics({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()}, epoch=epoch)
                results.update({f"dev_{k}_epoch{epoch}": v for k, v in dev_results.items()})

                tqdm.write("dev: " + str(dev_results))

                if dev_results["f1"] > best_dev_f1[0]:
                    logging.info('dev results {} {}'.format(dev_results, '**'))
                    logging.info('test results {} {}'.format(test_results, '**'))
                    # print(test_results, '**')
                    # if hasattr(model, "module"):
                    #     best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
                    # else:
                    #     best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                    best_dev_f1[0] = dev_results["f1"]
                    results["best_epoch"] = epoch
                else:
                    logging.info('dev results {}'.format(dev_results))
                    logging.info('test results {}'.format(test_results))
                model.train()

        trainer = Trainer(args, model=model, dataloader=train_dataloader, num_train_steps=num_train_steps,
                          step_callback=step_callback)  # 这儿相当于__init__
        trainer.train()  # 这儿相当于forward




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    # 1. load parameters
    args = parse_args()

    # # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG if args.local_rank in [-1, 0] else logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # Set seed
    set_seed(args)
    # ================================================================================================
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    #  load tokenizer: 使用的roberta-large
    tokenizer = RobertaTokenizer.from_pretrained(args.baseline_model_name)
    args.tokenizer = tokenizer
    model_archive = ModelArchive.load(args.checkpoint_file)
    args.model_config = model_archive.config

    experiment_logger = NullLogger()

    if args.local_rank in (-1, 0) and args.experiment_logger == "comet":
        experiment_logger = CometLogger(args)

    args.experiment = experiment_logger

    # load checkpoint
    model_weights = torch.load(os.path.join(args.checkpoint_file, 'pytorch_model.bin'), map_location=args.device)

    args.model_config.vocab_size += 1
    args.entity_vocab = model_archive.entity_vocab
    args.model_config.entity_vocab_size = 2
    args.model_config.k_entity_vocab_size = len(args.entity_vocab)

    # load model
    from data_utils import DatasetProcessor
    processor = DatasetProcessor()
    label_list = processor.get_label_list(args.data_dir)  # 9
    model = LukeForEntityTyping(args, num_labels=len(label_list))

    model.load_state_dict((model_weights, args), strict=False)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # ================================================================================================

    # ## Training
    # load dataset
    train_dataloader, _, features, _ = load_examples(args, fold="train")
    num_labels = len(features[0].labels)

    do_train(args, train_dataloader, model)


if __name__ == "__main__":
    main()