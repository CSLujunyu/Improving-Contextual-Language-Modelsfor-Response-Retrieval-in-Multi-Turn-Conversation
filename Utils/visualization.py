import numpy as np

def visualize_bad_cases(logits, input_file_path, output_file_path):

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    f = open(output_file_path, 'w', encoding='utf-8')
    ### example: (label \t context \t response)
    preds = np.argmax(logits, axis=-1)
    for logit, pred, example in zip(logits, preds, data):
        example = example.strip('\n').split('\t')
        label, context, response = example[0], ' [SEP] '.join(example[1:-1]), example[-1]
        f.write('Label:{}({:2.3f})\tPred:{}({:2.3f})\tContext: {}\tResponse: {}\n'.format(label, logit[int(label)], pred, logit[int(pred)], context, response))

    f.close()