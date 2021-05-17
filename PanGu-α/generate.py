"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor

def generate(model, origin_inputs, seq_length, end_token=50256):
    """
    TopK for text generation

    Inputs:
        model: the model for inferencing
        origin_inputs: the original inputs based on which the model will continue writing
        seq_length: seq_length for the model
        end_token: end of sentence token id

    Returns:
        outputs: the ids for the generated text
    """
    TOPK = 3
    seq_length = seq_length
    bs, valid_length = origin_inputs.shape
    pad_length = seq_length - origin_inputs.shape[-1]
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    print("input_ids is ", input_ids)
    while valid_length < seq_length:
        inputs = Tensor(input_ids, mstype.int32)
        probs, p_args = model.predict(inputs)
        probs = probs.asnumpy()[valid_length-1, :]
        p_args = p_args.asnumpy()[valid_length-1, :]

        p = probs
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        if p_args[target_index] == end_token or valid_length == seq_length-1:
            outputs = input_ids
            break
        input_ids[0][valid_length] = p_args[target_index]
        valid_length += 1
    length = np.sum(outputs != 0)
    outputs = outputs[0][:length]
    return outputs

