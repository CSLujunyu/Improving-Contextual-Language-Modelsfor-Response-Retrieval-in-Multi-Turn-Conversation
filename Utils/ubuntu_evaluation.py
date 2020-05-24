# coding=utf-8
import numpy as np

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def get_p_at_n_in_m(data, n, m, ind):
    pos_score = data[ind][0];
    curr = data[ind:ind + m];
    curr = sorted(curr, key=lambda x: x[0], reverse=True)

    if curr[n - 1][0] <= pos_score:
        return 1
    return 0

def evaluate(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip('\n').split('\t')
            if len(tokens) != 3:
                continue
            logits = [float(x) for x in tokens[1].split(',')]
            probs = softmax(logits)
            data.append([probs[-1], int(tokens[2])])

    assert len(data) % 10 == 0, print(len(data))

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0

    length = int(len(data) / 10)

    for i in range(0, length):
        ind = i * 10
        assert data[ind][1] == 1

        p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

    return {
        'R2@1': p_at_1_in_2 / length,
        'R10@1': p_at_1_in_10 / length,
        'R10@2': p_at_2_in_10 / length,
        'R10@5': p_at_5_in_10 / length
    }

if __name__ == '__main__':
    result = evaluate('/hdd/lujunyu/model/chatbert/check/logits_dev.txt')
