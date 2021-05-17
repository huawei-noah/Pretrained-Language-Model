"""
Create dataset for training and evaluting
"""


import os
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
import numpy as np

def get_input_data(input_ids, eod_id):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_token: the id for <EOD>

    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """

    seq_length = input_ids.shape[0] - 1
    attention_mask = np.tril(np.ones(shape=(seq_length, seq_length)))
    position_id = np.arange(seq_length)

    eod_index = position_id[input_ids[:-1] == eod_id]
    prev_index = 0
    for i in range(eod_index.size):
        index = eod_index[i]
        attention_mask[(index+1):, :(index+1)] = 0
        position_id[(index+1):] -= (index + 1 - prev_index)
        prev_index = index + 1
    return input_ids, position_id, attention_mask


def get_input_data_from_batch(input_ids, eod_id, rank, dis):
    """
    Generate position_id and attention_mask according to input_ids considering eod reset

    Inputs:
        input_ids: the input token ids
        eod_token: the id for <EOD>

    returns:
        input_ids: the input token ids
        position_id: the position ids cosidering eod reset
        attention_mask: the attention mask considering eod reset
    """
    rank = int(rank)
    input_ids = input_ids[rank * dis: (rank + 1) * dis]
    seq_length = 1024  # input_ids.shape[1] - 1

    batch_input_ids = input_ids
    batch_position_ids = np.ones((dis, seq_length))
    batch_attention_mask = np.ones((dis, seq_length, seq_length))
    for bs_i in range(0, len(input_ids)):
        local_ids = input_ids[bs_i]
        batch_attention_mask[bs_i] = np.tril(np.ones(shape=(seq_length, seq_length)))
        batch_position_ids[bs_i] = np.arange(seq_length)
        eod_index = batch_position_ids[bs_i, local_ids[:-1] == eod_id].astype(np.int32)
        prev_index = 0
        for i in range(eod_index.size):
            index = eod_index[i]
            batch_attention_mask[bs_i, (index + 1):, :(index + 1)] = 0
            batch_position_ids[bs_i, (index + 1):] -= (index + 1 - prev_index)
            prev_index = index + 1
    return batch_input_ids, batch_position_ids, batch_attention_mask

def create_dataset(batch_size, data_path, device_num=1, rank=0, drop=True, data_start_index=0, eod_reset=True, eod_id=9):
    """
    Create dataset

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_reset: whether enable position reset and attention mask reset
        eod_id: the id for <EOD>

    Returns:
        dataset: the dataset for training or evaluating
    """
    ds.config.set_seed(1)
    home_path = os.path.join(os.getcwd(), data_path)
    files = os.listdir(data_path)
        
    data = [
        os.path.join(home_path, name) for name in files
        if not name.endswith(".db")
    ]
    data.sort(key=lambda x: int(x[x.find("mindrecord")+10:]))
    print(data)
    dataset = ds.MindDataset(data[data_start_index:], columns_list=["input_ids"], shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float16)
    if eod_reset:
        map_func = (lambda input_ids: get_input_data(input_ids, eod_id))
        dataset = dataset.map(operations=map_func, input_columns=["input_ids"], output_columns=["input_ids", "position_id", "attention_mask"], column_order=["input_ids", "position_id", "attention_mask"])
        dataset = dataset.map(input_columns="position_id", operations=type_cast_op)
        dataset = dataset.map(input_columns="attention_mask", operations=type_cast_op_float)
    dataset = dataset.map(input_columns="input_ids", operations=type_cast_op)
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    dataset = dataset.repeat(1)
    return dataset


def create_dataset_dp(batch_size, data_path, device_num=1, rank=0, drop=True, data_start_index=0,
                   eod_id=9):
    """
    Create dataset using data parallel.

    Inputs:
        batch_size: batch size
        data_path: path of your MindRecord files
        device_num: total device number
        rank: current rank id
        drop: whether drop remainder
        eod_id: the id for <EOD>

    Returns:
        dataset: the dataset for training or evaluating
    """
    ds.config.set_seed(1)
    home_path = os.path.join(os.getcwd(), data_path)
    files = os.listdir(data_path)
    
    dis = int(batch_size / device_num)
    if dis < 1:
        raise ValueError("Batch size / device_num should be positive, but found {}".format(dis))

    data = [
        os.path.join(home_path, name) for name in files
        if not name.endswith(".db")
    ]
    data.sort(key=lambda x: int(x[x.find("mindrecord")+10:]))
    print(data)

    if data_start_index >= len(data):
        raise ValueError(f"data start index {data_start_index} is larger than dataset length {len(data)}")
    dataset = ds.MindDataset(data[data_start_index:], columns_list=["input_ids"], shuffle=False)
    type_cast_op = C.TypeCast(mstype.int32)
    type_cast_op_float = C.TypeCast(mstype.float16)

    map_func = (lambda input_ids: get_input_data_from_batch(input_ids, eod_id, rank, dis))
    dataset = dataset.batch(batch_size, drop_remainder=drop)
    dataset = dataset.map(operations=map_func, input_columns=["input_ids"],
                          output_columns=["input_ids", "position_id", "attention_mask"],
                          column_order=["input_ids", "position_id", "attention_mask"])
    dataset = dataset.map(input_columns="position_id", operations=type_cast_op)
    dataset = dataset.map(input_columns="attention_mask", operations=type_cast_op_float)

    dataset = dataset.map(input_columns="input_ids", operations=type_cast_op)
    dataset = dataset.repeat(1)
    return dataset
