# coding=utf-8
import random
import json
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_all_examples(self, data_dir):
        """Gets a collection of `InputExample`s."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = json.load(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class YelpProcessor(DataProcessor):
    def __init__(self):
        random.seed(42)

        with open("/hdd/lujunyu/dataset/yelp/yelp_academic_dataset_review.json", "r") as f:
            self.D = f.readlines()
        random.shuffle(self.D)

        ### Preprocess and Save
        self.get_all_examples("/hdd/lujunyu/dataset/yelp/yelp.all")

        with open("/hdd/lujunyu/dataset/yelp/labels.json", 'w', encoding='utf-8') as f:
            f.write(json.dumps(dict(zip(self.get_labels(), range(len(self.get_labels())))), indent=4, ensure_ascii=False))

    def get_all_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D, data_dir)

    def get_labels(self):
        """Task labels."""
        return ['1','2','3','4','5']

    def _create_examples(self, data, data_dir):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
            d = json.loads(d)
            review_id = d['review_id']
            text = d['text'].replace('\n', '').replace('\\', '').replace('\r', '').replace('\t', '').lower()
            label = int(d['stars'])
            examples.append('{}\t{}\t{:d}\n'.format(review_id, text, label))

        with open(data_dir, 'w', encoding='utf-8') as f:
            f.writelines(examples)

        logger.info('Yelp Data size: {:d}'.format(len(examples)))
        return examples

if __name__ == '__main__':
    yelp_p = YelpProcessor()