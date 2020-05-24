# coding=utf-8
import os
import random
import json
import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class DoubanDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, turn_data = data[0], data[1:]

        if len(turn_data[-1].split()) > self.max_seq_length - 1:
            turn_data[-1] = ' '.join(turn_data[-1].split()[:(self.max_seq_length - 1)//2])

        input_text = []
        for t in turn_data:
            tokens_t = self.tokenizer.tokenize(t)
            input_text = input_text + tokens_t + ['[SEP]']

        if len(input_text) > self.max_seq_length - 1:
            input_text = input_text[-(self.max_seq_length - 1):]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        seg_flag = 0

        for token in input_text:
            tokens.append(token)
            segment_ids.append(seg_flag)
            if token == '[SEP]':
                seg_flag = 1 - seg_flag

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

class DoubanDatasetForSP(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, input_context, input_candidate = data[0], ' '.join(data[1:-1]), data[-1]

        tokens_a = self.tokenizer.tokenize(input_context)

        tokens_b = self.tokenizer.tokenize(input_candidate)

        if len(tokens_a + tokens_b) > self.max_seq_length - 3:
            if len(tokens_a) > self.max_seq_length - 3 and len(tokens_b) > self.max_seq_length - 3:
                tokens_a = tokens_a[0:(self.max_seq_length - 3)//2]
                tokens_b = tokens_b[0:(self.max_seq_length - 3 - len(tokens_a))]
            else:
                if len(tokens_a) > self.max_seq_length - 3:
                    tokens_a = tokens_a[0:(self.max_seq_length - 3 - len(tokens_b))]
                else:
                    tokens_b = tokens_b[0:(self.max_seq_length - 3 - len(tokens_a))]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length, print(len(input_ids), tokens, len(tokens_a), len(tokens_b))
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

class UbuntuDataset(Dataset):
    """Ubuntu dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the txt file.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, turn_data = data[0], data[1:]

        if len(turn_data[-1].split()) > self.max_seq_length - 1:
            turn_data[-1] = ' '.join(turn_data[-1].split()[:(self.max_seq_length - 1)//2])

        input_text = []
        for t in turn_data:
            tokens_t = self.tokenizer.tokenize(t)
            input_text = input_text + tokens_t + ['[SEP]']

        if len(input_text) > self.max_seq_length - 1:
            input_text = input_text[-(self.max_seq_length - 1):]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        seg_flag = 0

        for token in input_text:
            tokens.append(token)
            segment_ids.append(seg_flag)
            if token == '[SEP]':
                seg_flag = 1 - seg_flag

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

class UbuntuDatasetForWrongSpeakerTesting(Dataset):
    """Ubuntu dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the txt file.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, turn_data = data[0], data[1:]

        if len(turn_data[-1].split()) > self.max_seq_length - 1:
            turn_data[-1] = ' '.join(turn_data[-1].split()[:(self.max_seq_length - 1)//2])

        input_text = []
        for t in turn_data:
            tokens_t = self.tokenizer.tokenize(t)
            input_text = input_text + tokens_t + ['[SEP]']

        if len(input_text) > self.max_seq_length - 1:
            input_text = input_text[-(self.max_seq_length - 1):]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        seg_flag = 0
        current_turn = 1
        num_turns = input_text.count('[SEP]')
        for token in input_text:
            tokens.append(token)
            segment_ids.append(seg_flag)
            if token == '[SEP]':
                current_turn += 1
                if current_turn != num_turns:
                    seg_flag = 1 - seg_flag

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

class UbuntuDatasetForRoberta(Dataset):
    """Ubuntu dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the txt file.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, turn_data = data[0], data[1:]

        if len(turn_data[-1].split()) > self.max_seq_length - 1:
            turn_data[-1] = ' '.join(turn_data[-1].split()[:(self.max_seq_length - 1)//2])

        input_ids = [0]
        for t in turn_data:
            tokens_t = self.tokenizer.encode(t, add_special_tokens=False)
            if input_ids == [0]:
                input_ids = input_ids + tokens_t + [2]
            else:
                input_ids = input_ids + [2] + tokens_t + [2]

        if len(input_ids) > self.max_seq_length - 1:
            input_ids = input_ids[-(self.max_seq_length - 1):]
            input_ids = [0] + input_ids

        seg_flag = 0
        seg_num = 0
        segment_ids = []
        for token_id in input_ids:
            if token_id == 0 or token_id == 2:
                if seg_flag == 1:
                    segment_ids.append(seg_num)
                    seg_num = 1 - seg_num
                    seg_flag = 0
                else:
                    seg_flag = 1
                    segment_ids.append(seg_num)
            else:
                segment_ids.append(seg_num)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(1)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )
        return sample

class UbuntuDatasetForRobertaSP(Dataset):
    """Ubuntu dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the txt file.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, input_context, input_candidate = data[0], ' '.join(data[1:-1]), data[-1]

        tokens_a = self.tokenizer.encode(input_context, add_special_tokens=False)
        tokens_b = self.tokenizer.encode(input_candidate, add_special_tokens=False)

        if len(tokens_a + tokens_b) > self.max_seq_length - 4:
            if len(tokens_a) > self.max_seq_length - 4 and len(tokens_b) > self.max_seq_length - 4:
                tokens_a = tokens_a[0:(self.max_seq_length - 4) // 2]
                tokens_b = tokens_b[0:(self.max_seq_length - 4 - len(tokens_a))]
            else:
                if len(tokens_a) > self.max_seq_length - 4:
                    tokens_a = tokens_a[0:(self.max_seq_length - 4 - len(tokens_b))]
                else:
                    tokens_b = tokens_b[0:(self.max_seq_length - 4 - len(tokens_a))]

        input_ids = []
        segment_ids = []
        input_ids.append(0)
        segment_ids.append(0)
        for token in tokens_a:
            input_ids.append(token)
            segment_ids.append(0)
        input_ids.append(2)
        input_ids.append(2)
        segment_ids.append(0)
        segment_ids.append(0)

        for token in tokens_b:
            input_ids.append(token)
            segment_ids.append(0)
        input_ids.append(2)
        segment_ids.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(1)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

class UbuntuDatasetForSP(Dataset):
    """Ubuntu dataset."""

    def __init__(self, file_path, tokenizer, max_seq_length=512):
        """
        Args:
            file_path (string): Path to the txt file.
        """

        self.max_seq_length = max_seq_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        data = self.data[idx].strip('\n').split('\t')

        label, input_context, input_candidate = data[0], ' '.join(data[1:-1]), data[-1]

        tokens_a = self.tokenizer.tokenize(input_context)

        tokens_b = self.tokenizer.tokenize(input_candidate)

        if len(tokens_a + tokens_b) > self.max_seq_length - 3:
            if len(tokens_a) > self.max_seq_length - 3 and len(tokens_b) > self.max_seq_length - 3:
                tokens_a = tokens_a[0:(self.max_seq_length - 3) // 2]
                tokens_b = tokens_b[0:(self.max_seq_length - 3 - len(tokens_a))]
            else:
                if len(tokens_a) > self.max_seq_length - 3:
                    tokens_a = tokens_a[0:(self.max_seq_length - 3 - len(tokens_b))]
                else:
                    tokens_b = tokens_b[0:(self.max_seq_length - 3 - len(tokens_a))]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        label_id = int(label)

        sample = (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(input_mask, dtype=torch.long),
            torch.tensor(segment_ids, dtype=torch.long),
            torch.tensor(label_id, dtype=torch.long)
        )

        return sample

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    d = UbuntuDataset(
        file_path="/hdd/lujunyu/dataset/multi_turn_corpus/ubuntu/train.txt",
        tokenizer=tokenizer
    )
    for item in d:
     pass