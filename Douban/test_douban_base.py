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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import argparse
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, SequentialSampler
from Utils.DataLoader import UbuntuDatasetBase
from pytorch_transformers import BertTokenizer

from Model.modeling_bert import BertConfig, BertForSequenceClassification
from Utils.ubuntu_evaluation import evaluate

reverse_order = False
sa_step = False
BERT_BASE_DIR = '/hdd/lujunyu/dataset/bert/chinese_L-12_H-768_A-12/'


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/hdd/lujunyu/dataset/ubuntu/',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_file",
                        default=BERT_BASE_DIR + 'bert_config.json',
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--task_name",
                        default='ubuntu',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--vocab_file",
                        default=BERT_BASE_DIR + 'vocab.txt',
                        type=str,
                        required=False,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--output_dir",
                        default='/hdd/lujunyu/model/ubuntu_base/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--init_checkpoint",
                        default='/hdd/lujunyu/model/ubuntu_base/',
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")

    ## Other parameters
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    parser.add_argument("--eval_batch_size",
                        default=100,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    bert_config = BertConfig.from_json_file(args.bert_config_file)
    tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_BASE_DIR, 'vocab.txt'),
                                              do_lower_case=args.do_lower_case)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
            args.max_seq_length, bert_config.max_position_embeddings))

    test_dataset = UbuntuDatasetBase(
        file_path=os.path.join(args.data_dir, "test.txt"),
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size,
                                                sampler=SequentialSampler(test_dataset))

    model = BertForSequenceClassification(bert_config)
    state_dict = torch.load(os.path.join(args.init_checkpoint, "model.pt"), map_location='cpu')

    new_state_dict = OrderedDict()
    for name, param in list(state_dict.items()):
        name = name.strip('module.')
        new_state_dict[name] = param

    model.load_state_dict(new_state_dict)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    logger.info("***** Running testing *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    f = open(os.path.join(args.output_dir, 'logits_test.txt'), 'w', encoding='utf-8')

    model.eval()
    test_loss = 0
    nb_test_steps, nb_test_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in test_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_test_loss, logits = model(input_ids, segment_ids, input_mask, label_ids)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()

        for logit, label in zip(logits, label_ids):
            logit = '{},{}'.format(logit[0], logit[1])
            f.write('_\t{}\t{}\n'.format(logit, label))

        test_loss += tmp_test_loss.mean().item()

        nb_test_examples += input_ids.size(0)
        nb_test_steps += 1

    f.close()

    result = evaluate(os.path.join(args.output_dir, 'logits_test.txt'))
    test_loss = test_loss / nb_test_steps

    output_eval_file = os.path.join(args.output_dir, "results_test.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Test results *****")
        logger.info('test_loss: %s', test_loss)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == "__main__":
    main()
