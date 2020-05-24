import sys
import numpy as np
from sklearn.metrics import average_precision_score

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[2] for s_d in sort_data]
    select_lable = sort_lable[:k]
    if sort_lable.count(1) != 0:
        return 1.0 * select_lable.count(1) / sort_lable.count(1)


def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[1], reverse=True)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return r_1, r_2, r_5


def evaluate(file_path):
    sum_r_1 = 0
    sum_r_2 = 0
    sum_r_5 = 0

    i = 0
    total_num = 0
    data = []
    useful_example = 0

    with open(file_path, 'r') as infile:
        for i, line in enumerate(infile):

            tokens = line.strip().split('\t')
            logits = [float(x) for x in tokens[1].split(',')]
            probs = softmax(logits)
            data.append([tokens[0], probs[-1], int(tokens[2])])

            useful_example += int(tokens[2])

            if i % 10 == 9:

                assert data[0][-1] == 1

                if useful_example == 0:
                    data = []
                    continue
                r_1, r_2, r_5 = evaluation_one_session(data)
                total_num += 1
                sum_r_1 += r_1
                sum_r_2 += r_2
                sum_r_5 += r_5
                data = []
                useful_example = 0

    print('total num: %s' %total_num)

    return {
               'R10@1':1.0 * sum_r_1 / total_num,
               'R10@2':1.0 * sum_r_2 / total_num,
               'R10@5':1.0 * sum_r_5 / total_num
    }


if __name__ == '__main__':
    result = evaluate('/home/lujunyu/repository/chatBERT/Data/logits_test.txt')
    print(result)