# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from Utils.DataLoader import UbuntuDatasetForRoberta
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from pytorch_transformers import AdamW, WarmupLinearSchedule
from Utils.ubuntu_evaluation import evaluate
from Utils.visualization import visualize_bad_cases


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs==labels)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/hdd/lujunyu/dataset/multi_turn_corpus/ubuntu/',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default='ubuntu',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='/hdd/lujunyu/model/chatbert/ubuntu_roberta_si_aug/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--dialog_augmentation_path",
                        default='/hdd/lujunyu/dataset/multi_turn_corpus/ubuntu/ubuntu_train_augment_largethan_13.txt',
                        type=str,
                        help="Whether to use augmentation")
    ## Other parameters
    parser.add_argument("--init_model_name",
                        default='roberta-base',
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size",
                        default=600,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=200,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay",
                        default=1e-3,
                        type=float,
                        help="weight_decay")
    parser.add_argument("--save_checkpoints_steps",
                        default=3000,
                        type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=10,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = RobertaConfig.from_pretrained(args.init_model_name, num_labels=2)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = RobertaTokenizer.from_pretrained(args.init_model_name, do_lower_case=args.do_lower_case)
    if args.dialog_augmentation_path:
        train_dataset = UbuntuDatasetForRoberta(
            file_path=args.dialog_augmentation_path,
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer
        )
    else:
        train_dataset = UbuntuDatasetForRoberta(
            file_path=os.path.join(args.data_dir, "train.txt"),
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer
        )
    eval_dataset = UbuntuDatasetForRoberta(
        file_path=os.path.join(args.data_dir, "test.txt"),
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
                                                sampler=RandomSampler(train_dataset), num_workers=4)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.eval_batch_size,
                                                sampler=SequentialSampler(eval_dataset), num_workers=8)

    model = RobertaForSequenceClassification.from_pretrained(args.init_model_name, config=bert_config)
    model.config.type_vocab_size = 2
    single_emb = model.roberta.embeddings.token_type_embeddings
    model.roberta.embeddings.token_type_embeddings = torch.nn.Embedding(2, single_emb.embedding_dim)
    model.roberta.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))
    model.to(device)

    num_train_steps = None
    if args.do_train:
        num_train_steps = int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        # remove pooler, which is not used thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay}, {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_train_steps)
    else:
        optimizer = None
        scheduler = None

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    best_metric = 0.0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()    # We have accumulated enought gradients
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % args.save_checkpoints_steps == 0:
                    model.eval()
                    f = open(os.path.join(args.output_dir, 'logits_test.txt'), 'w')
                    eval_loss = 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    logits_all = []
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)

                        with torch.no_grad():
                            tmp_eval_loss, logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)

                        logits = logits.detach().cpu().numpy()
                        logits_all.append(logits)
                        label_ids = label_ids.cpu().numpy()

                        for logit, label in zip(logits, label_ids):
                            logit = '{},{}'.format(logit[0], logit[1])
                            f.write('_\t{}\t{}\n'.format(logit, label))

                        eval_loss += tmp_eval_loss.mean().item()

                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    f.close()
                    logits_all = np.concatenate(logits_all,axis=0)
                    eval_loss = eval_loss / nb_eval_steps

                    result = evaluate(os.path.join(args.output_dir, 'logits_test.txt'))
                    result.update({'eval_loss': eval_loss})

                    output_eval_file = os.path.join(args.output_dir, "eval_results_test.txt")
                    with open(output_eval_file, "a") as writer:
                        logger.info("***** Eval results *****")
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))

                    ### Save the best checkpoint
                    if best_metric < result['R10@1'] + result['R10@2']:
                        try:  ### Remove 'module' prefix when using DataParallel
                            state_dict = model.module.state_dict()
                        except AttributeError:
                            state_dict = model.state_dict()
                        torch.save(state_dict, os.path.join(args.output_dir, "model.pt"))
                        best_metric = result['R10@1'] + result['R10@2']
                        logger.info('Saving the best model in {}'.format(os.path.join(args.output_dir, "model.pt")))

                        ### visualize bad cases of the best model
                        logger.info('Saving Bad cases...')
                        visualize_bad_cases(
                            logits=logits_all,
                            input_file_path=os.path.join(args.data_dir, 'test.txt'),
                            output_file_path=os.path.join(args.output_dir, 'test_bad_cases.txt')
                        )

                    model.train()

if __name__ == "__main__":
    main()
