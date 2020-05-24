import sys
import numpy as np
from sklearn.metrics import average_precision_score

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def mean_average_precision(sort_data):
    # to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][2] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index + 1)
    return sum_precision / count_1


def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[2] for s_d in sort_data]
    if 1 in sort_lable:
        return 1.0 / (1 + sort_lable.index(1))


def precision_at_position_1(sort_data):
    if sort_data[0][2] == 1:
        return 1
    else:
        return 0


def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[2] for s_d in sort_data]
    select_lable = sort_lable[:k]
    if sort_lable.count(1) != 0:
        return 1.0 * select_lable.count(1) / sort_lable.count(1)


def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[1], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5


def evaluate(file_path):
    sum_m_a_p = 0
    sum_m_r_r = 0
    sum_p_1 = 0
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
                if useful_example == 0:
                    data = []
                    continue
                m_a_p, m_r_r, p_1, r_1, r_2, r_5 = evaluation_one_session(data)
                total_num += 1
                sum_m_a_p += m_a_p
                sum_m_r_r += m_r_r
                sum_p_1 += p_1
                sum_r_1 += r_1
                sum_r_2 += r_2
                sum_r_5 += r_5
                data = []
                useful_example = 0

    print('total num: %s' %total_num)

    return {
               'MAP':1.0 * sum_m_a_p / total_num,
               'MRR':1.0 * sum_m_r_r / total_num,
               'P@1':1.0 * sum_p_1 / total_num,
               'R10@1':1.0 * sum_r_1 / total_num,
               'R10@2':1.0 * sum_r_2 / total_num,
               'R10@5':1.0 * sum_r_5 / total_num
    }


if __name__ == '__main__':
    result = evaluate('/hdd/lujunyu/model/douban/logits.txt')