import os

import numpy as np

from .converter import CTCLabelConverter

character = '0123456789!ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюяЂђЃѓЄєІіЇїЈјЉљЊњЋћЌќЎўЏџҐґҒғҚқҮүҲҳҶҷӀӏӢӣӨөӮӯ'
separator_list = {'ru': []}
dict_list = {'ru': f"{os.path.dirname(os.path.abspath(__file__))}/ru.txt"}

converter = CTCLabelConverter(character, separator_list, dict_list)

def softmax(x, axis=None):
    exp_x = np.exp(x)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))


# здесь batch_size строго 1
def postprocess(preds, decoder='greedy', beamWidth=5):
    result = []

    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = np.full(1, preds.shape[1], dtype=np.int32)

    preds_prob = softmax(preds, axis=2)
    pred_norm = preds_prob.sum(axis=2)
    preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)

    if decoder == 'greedy':
        # Select max probabilty (greedy decoding) then decode index to character
        preds_index = preds_prob.argmax(axis=2)
        preds_index = preds_index.ravel()
        preds_str = converter.decode_greedy(preds_index, preds_size)
    elif decoder == 'beamsearch':
        preds_str = converter.decode_beamsearch(preds_prob, beamWidth=beamWidth)
    elif decoder == 'wordbeamsearch':
        preds_str = converter.decode_wordbeamsearch(preds_prob, beamWidth=beamWidth)

    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    preds_max_prob = []
    for v,i in zip(values, indices):
        max_probs = v[i!=0]
        if len(max_probs)>0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))

    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)
        result.append([pred, confidence_score])

    return result