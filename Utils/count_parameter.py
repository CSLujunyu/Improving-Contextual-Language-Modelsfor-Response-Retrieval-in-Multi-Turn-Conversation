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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
from pytorch_transformers import RobertaConfig, RobertaForSequenceClassification
from pytorch_transformers import BertConfig, BertForSequenceClassification
from transformers import AlbertConfig, AlbertTokenizer, AlbertForSequenceClassification

def main():

    bert_base_config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
    bert_base_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=bert_base_config)
    count = 0
    for name, param in bert_base_model.named_parameters():
        if param.requires_grad:
            size = 1
            for s in param.data.size():
                size = s * size
            count += size
    print('The total number of parameters in bert_base_uncased: ', count)

    roberta_config = RobertaConfig.from_pretrained('roberta-base', num_labels=2)
    roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base',config=roberta_config)
    count = 0
    for name, param in roberta_model.named_parameters():
        if param.requires_grad:
            size = 1
            for s in param.data.size():
                size = s * size
            count += size
    print('The total number of parameters in roberta: ', count)

    albert_config = AlbertConfig.from_pretrained('albert-base-v2', num_labels=2)
    albert_model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', config=albert_config)
    count = 0
    for name, param in albert_model.named_parameters():
        if param.requires_grad:
            size = 1
            for s in param.data.size():
                size = s * size
            count += size
    print('The total number of parameters in albert: ', count)

if __name__ == "__main__":
    main()
