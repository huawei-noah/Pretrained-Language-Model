"""PANGUALPHA model"""

import math
import numpy as np
import os
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Normal, TruncatedNormal
from mindspore.ops import operations as P
from mindspore.ops import functional as F

class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, dp=4, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = Parameter(initializer('ones', normalized_shape), name="gamma", parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape), name="beta", parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True).shard(((dp, 1, 1),))
        self.square = P.Square().shard(((dp, 1, 1),))
        self.sqrt = P.Sqrt().shard(((dp, 1, 1),))
        self.sub1 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.sub2 = P.Sub().shard(((dp, 1, 1), (dp, 1, 1)))
        self.add = P.TensorAdd().shard(((dp, 1, 1), ()))
        self.eps = eps
        self.mul = P.Mul().shard(((dp, 1, 1), (1,)))
        self.add2 = P.TensorAdd().shard(((dp, 1, 1), (1,)))
        self.real_div = P.RealDiv().shard(((dp, 1, 1), (dp, 1, 1)))
    def construct(self, x):
        mean = self.mean(x, -1)
        diff = self.sub1(x, mean)
        variance = self.mean(self.square(diff), -1)
        variance_eps = self.sqrt(self.add(variance, self.eps))
        output = self.real_div(diff, variance_eps)
        output = self.add2(self.mul(output, self.gamma), self.beta)
        return output


class Mapping(nn.Cell):
    """
    A mapping function with a 3d input
    Args:
        input_size: the size of the last dimension of the input tensor
        output_size: the desired size of the last dimension of the output tensor
        dtype: the compute datatype
        scale: the scale factor for initialization
    Inputs:
        x: the 3d input
    Returns:
        output: Tensor, a 3d tensor after projection
    """
    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size]),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ]),
                              name="mapping_bias",
                              parallel_optimizer=False)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        #self.cast.add_prim_attr("_side_effect", True)
        self.add = P.TensorAdd().shard(((config.dp, 1), (1,)))
        self.matmul = P.MatMul().shard(
            ((config.dp, config.mp), (config.mp, 1)))
        self.matmul.add_prim_attr("recompute_comm_op", False)

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output


class Mapping_output(nn.Cell):
    def __init__(self, config, input_size, output_size, scale=1.0):
        super(Mapping_output, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.weight = Parameter(initializer(Normal(sigma=0.02 * scale),
                                            [input_size, output_size]),
                                name="mapping_weight")
        self.bias = Parameter(initializer("zeros", [
            output_size,
        ]),
                              name="mapping_bias")
        self.dtype = config.compute_dtype
        self.cast = P.Cast()
        #self.cast.add_prim_attr("_side_effect", True)
        self.add = P.TensorAdd().shard(((config.dp, config.mp), (config.mp,)))
        self.matmul = P.MatMul().shard(((config.dp, 1), (1, config.mp)))

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.output_size,)
        x = P.Reshape()(x, (-1, self.input_size))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        x = self.add(x, self.cast(self.bias, self.dtype))
        output = P.Reshape()(x, out_shape)
        return output


class Output(nn.Cell):
    """
    The output mapping module for each layer
    Args:
        config(PANGUALPHAConfig): the config of network
        scale: scale factor for initialization
    Inputs:
        x: output of the self-attention module
    Returns:
        output: Tensor, the output of this layer after mapping
    """
    def __init__(self, config, scale=1.0):
        super(Output, self).__init__()
        input_size = config.embedding_size
        output_size = config.embedding_size * config.expand_ratio
        self.mapping = Mapping_output(config, input_size, output_size)
        self.projection = Mapping(config, output_size, input_size, scale)
        self.activation = nn.GELU()
        self.activation.gelu.shard(((config.dp, 1, config.mp),))
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

    def construct(self, x):
        hidden = self.activation(self.mapping(x))
        output = self.projection(hidden)
        output = self.dropout(output)
        return output


class AttentionMask(nn.Cell):
    """
    Get the attention matrix for self-attention module
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
    """
    def __init__(self, config):
        super(AttentionMask, self).__init__()
        self.reshape = P.Reshape()
        self.mul = P.BatchMatMul().shard(
            ((config.dp, 1, 1), (config.dp, 1, 1)))  # yzz: use 64, 1, 1?
        self.expand_dim = P.ExpandDims().shard(((1, 1),))
        ones = np.ones(shape=(config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), mstype.float32)
        self.multiply = P.Mul().shard(((config.dp, 1, 1), (1, 1, 1)))

    def construct(self, input_mask):
        input_shape = P.Shape()(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1])
        shape_left = input_shape + (1,)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)
        attention_mask = self.mul(mask_left, mask_right)
        lower_traiangle = self.expand_dim(self.lower_triangle_mask, 0)
        attention_mask = self.multiply(
            attention_mask, lower_traiangle)  #bs seq_length seq_length
        return attention_mask


class EmbeddingLookupPipeline(nn.Cell):
    """
    The embedding lookup table for vocabulary
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
    Returns:
        output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
        self.embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(EmbeddingLookupPipeline, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.word_emb_dp:
            self.gather = P.GatherV2().shard(((1, 1), (config.dp, 1)))
        else:
            self.gather = P.GatherV2().shard(((config.mp, 1), (1, 1)))
            self.gather.add_prim_attr("repeated_calc_num_direction", "left")
            if config.forward_reduce_scatter:
                self.gather.add_prim_attr("forward_type", "ReduceScatter")
        self.gather.add_prim_attr("begin", 0)
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids, table):
        output = self.gather(table, input_ids, 0)
        return output


class EmbeddingLookup(nn.Cell):
    def __init__(self, config):
        """
        The embedding lookup table for vocabulary
        Args:
            config(PANGUALPHAConfig): the config of network
        Inputs:
            input_ids: the tokenized inputs with datatype int32
        Returns:
            output: Tensor, the embedding vector for the input with shape (batch_size, seq_length, embedding_size)
            self.embedding_table: Tensor, the embedding table for the vocabulary
        """
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        if config.load_ckpt_path:
            # Loading the embedding table from the ckpt path:
            embedding_path = os.path.join(config.load_ckpt_path, 'word_embedding.npy')
            if os.path.exists(embedding_path):
                e_table = np.load(embedding_path)
                e_table = Tensor(e_table, mstype.float32)
                self.embedding_table = Parameter(e_table, name="embedding_table")
            else:
                raise ValueError(f"{embedding_path} file not exits, please check whether word_embedding file exist.")
        else:
            self.embedding_table = Parameter(initializer(
                Normal(0.02), [self.vocab_size, self.embedding_size]),
                                             name="embedding_table")
        if config.word_emb_dp:
            self.gather = P.GatherV2().shard(((1, 1), (config.dp, 1)))
        else:
            self.gather = P.GatherV2().shard(((config.mp, 1), (1, 1)))
            self.gather.add_prim_attr("repeated_calc_num_direction", "left")
            if config.forward_reduce_scatter:
                self.gather.add_prim_attr("forward_type", "ReduceScatter")
        self.shape = (-1, config.seq_length, config.embedding_size)

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table


class Attention(nn.Cell):
    """
    Self-Attention module for each layer

    Args:
        config(PANGUALPHAConfig): the config of network
        scale: scale factor for initialization
        layer_idx: current layer index
    """
    def __init__(self, config, scale=1.0, layer_idx=None):
        super(Attention, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.projection = Mapping(config, config.embedding_size,
                                  config.embedding_size, scale)
        self.transpose = P.Transpose().shard(((config.dp, 1, config.mp, 1),))
        self.merger_head_transpose = P.Transpose().shard(
            ((config.dp, config.mp, 1, 1),))
        self.reshape = P.Reshape()
        self.n_head = config.num_heads
        self.size_per_head = config.embedding_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=mstype.float32)
        self.batch_matmul = P.BatchMatMul().shard(
            ((config.dp, config.mp, 1, 1), (config.dp, config.mp, 1, 1)))
        self.scale = scale
        self.real_div = P.RealDiv().shard(((config.dp, config.mp, 1, 1), ()))
        self.sub = P.Sub().shard(((1,), (config.dp, 1, 1, 1))).add_prim_attr("_side_effect", True)
        self.mul = P.Mul().shard(((config.dp, 1, 1, 1), (1,))).add_prim_attr("_side_effect", True)
        self.add = P.TensorAdd().shard(
            ((config.dp, 1, 1, 1), (config.dp, config.mp, 1, 1)))
        if self.scale:
            self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        if layer_idx is not None:
            self.coeff = math.sqrt(layer_idx * math.sqrt(self.size_per_head))
            self.coeff = Tensor(self.coeff)
        self.use_past = config.use_past
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))
        self.prob_dropout = nn.Dropout(1 - config.dropout_rate)
        self.prob_dropout.dropout_gen_mask.shard(
            ((config.dp, config.mp, 1, 1),))
        self.prob_dropout.dropout_do_mask.shard(
            ((config.dp, config.mp, 1, 1),))
        self.softmax = nn.Softmax()
        self.softmax.softmax.shard(((config.dp, config.mp, 1),))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))

        self.dense1 = nn.Dense(config.embedding_size,
                               config.embedding_size).to_float(
            config.compute_dtype)
        self.dense1.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense1.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense2 = nn.Dense(config.embedding_size,
                               config.embedding_size).to_float(
            config.compute_dtype)
        self.dense2.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense2.bias_add.shard(((config.dp, config.mp), (config.mp,)))
        self.dense3 = nn.Dense(config.embedding_size,
                               config.embedding_size).to_float(
            config.compute_dtype)
        self.dense3.matmul.shard(((config.dp, 1), (config.mp, 1)))
        self.dense3.bias_add.shard(((config.dp, config.mp), (config.mp,)))

    def construct(self, x, attention_mask, layer_past=None):
        """
        self-attention

        Inputs:
            x: output of previous layer
            attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
            layer_past: the previous feature map

        Returns:
            output: Tensor, the output logit of this layer
            layer_present: Tensor, the feature map of current layer
        """

        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        query = self.dense1(x)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(
            F.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        key = self.transpose(
            F.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        value = self.transpose(
            F.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = P.Pack()([self.transpose(key, (0, 1, 3, 2)), value])
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present

    def split_heads(self, x, transpose):
        """
        split 3d tensor to 4d and switch certain axes
        Inputs:
            x: input tensor
            transpose: tuple, the transpose sequence
        Returns:
            x_transpose: the 4d output
        """
        x_size = P.Shape()(x)
        new_x_shape = x_size[:-1] + (self.n_head, self.size_per_head)
        x = self.reshape(x, new_x_shape)
        x_transpose = self.transpose(x, transpose)
        return x_transpose

    def merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Returns:
            x_merge: the 3d output
        """
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))  #bs, seq_length, head, size_per_head
        x_shape = P.Shape()(x)
        new_shape = x_shape[:-2] + (x_shape[-2] * x_shape[-1],)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length
        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        Returns:
            weighted_values: Tensor, the weighted sum scores
        """
        if not self.scale:
            query = query / F.cast(self.coeff, F.dtype(query))
            key = key / F.cast(self.coeff, F.dtype(key))

        score = self.batch_matmul(query, key)
        if self.scale:
            score = self.real_div(
                score,
                P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, mstype.float32)
        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))

        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)

        shape = F.shape(attention_scores)
        attention_probs = self.softmax(
            F.reshape(attention_scores,
                      (shape[0], -1, shape[-1])))
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = F.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values


class Block(nn.Cell):
    """
    The basic block of PANGUALPHA network
    Args:
        config(PANGUALPHAConfig): the config of network
        layer_idx: current layer index
    Inputs:
        x: the output of previous layer(input_ids for the first layer)
        attention_mask: the attention mask matrix with shape (batch_size, 1, seq_length, seq_length)
        layer_past: the previous feature map
    Returns:
        output: Tensor, the output logit of this layer
        layer_present: Tensor, the feature map of current layer
    """
    def __init__(self, config, layer_idx):
        super(Block, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        if config.self_layernorm:
            self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
            self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        else:
            self.layernorm1 = nn.LayerNorm(
                (config.embedding_size,)).to_float(mstype.float32)
            self.layernorm1.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
            self.layernorm2 = nn.LayerNorm(
                (config.embedding_size,)).to_float(mstype.float32)
            self.layernorm2.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        
        self.attention = Attention(config, scale, layer_idx)
        self.output = Output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.add_last = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.add_last.recompute(False)
        self.dtype = config.compute_dtype

    def construct(self, x, input_mask, layer_past=None):
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, input_mask,
                                                  layer_past)
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.add_last(output_x, mlp_logit)
        else:
            output = self.add_last(x, mlp_logit)
        return output, layer_present


class PANGUALPHA_EmbeddingPipeLine(nn.Cell):
    def __init__(self, config):
        super(PANGUALPHA_EmbeddingPipeLine, self).__init__()
        self.word_embedding = EmbeddingLookupPipeline(config)
        self.position_embedding = nn.Embedding(config.seq_length,
                                               config.embedding_size,
                                               embedding_table=Normal(0.02))
        self.position_embedding.gather.shard(((1, 1), (config.dp ,)))
        self.position_embedding.expand.shard(((config.dp, 1),))
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

    def construct(self, input_ids, table, input_position):
        input_embedding = self.word_embedding(input_ids, table)
        position_embedding = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        return hidden_states


class PANGUALPHA_Mask(nn.Cell):
    def __init__(self, config):
        super(PANGUALPHA_Mask, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.dtype = config.compute_dtype
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))
    def construct(self, input_mask, attention_mask):
        attention_mask = self.expand_dims(attention_mask, 1)
        return attention_mask


class QueryLayerAttention(Attention):
    def construct(self, x, query_hidden_state, attention_mask, layer_past=None):
        original_shape = F.shape(x)
        x = F.reshape(x, (-1, original_shape[-1]))
        query_hidden_state = F.reshape(query_hidden_state, (-1, original_shape[-1]))
        query = self.dense1(query_hidden_state)
        key = self.dense2(x)
        value = self.dense3(x)
        query = self.transpose(
            F.reshape(
                query,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        key = self.transpose(
            F.reshape(
                key, (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        value = self.transpose(
            F.reshape(
                value,
                (-1, original_shape[1], self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if self.use_past:
            past_value = layer_past[1]
            past_key = self.transpose(layer_past[0], (0, 1, 3, 2))
            key = self.concat_k((past_key, key))
            value = self.concat_v(past_value, value)
        layer_present = P.Pack()([self.transpose(key, (0, 1, 3, 2)), value])
        attention = self._attn(query, key, value, attention_mask)
        attention_merge = self.merge_heads(attention)
        output = self.projection(attention_merge)
        output = self.dropout(output)
        return output, layer_present

class QueryLayer(nn.Cell):
    def __init__(self, config):
        super(QueryLayer, self).__init__()
        scale = 1 / math.sqrt(2.0 * config.num_layers)
        self.layernorm1 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm2 = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        self.layernorm1.gamma.parallel_optimizer = False
        self.layernorm1.beta.parallel_optimizer = False
        self.attention = QueryLayerAttention(config, scale)
        self.layernorm2.gamma.parallel_optimizer = False
        self.layernorm2.beta.parallel_optimizer = False
        self.output = Output(config, scale)
        self.post_layernorm_residual = config.post_layernorm_residual
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))

        self.last_add = P.TensorAdd().shard(
            ((config.dp, 1, 1), (config.dp, 1,
                                 1))).add_prim_attr("recompute", False)
        self.dtype = config.compute_dtype

    def construct(self, x, query_hidden_state, input_mask, layer_past=None):
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x,
                                                  query_hidden_state,
                                                  input_mask,
                                                  layer_past)
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        if self.post_layernorm_residual:
            output = self.last_add(output_x, mlp_logit)
        else:
            output = self.last_add(x, mlp_logit)
        return output, layer_present

class PANGUALPHA_ModelPipeline(nn.Cell):
    """
    The backbone of PANGUALPHA network
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map
    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """
    def __init__(self, config):
        super(PANGUALPHA_ModelPipeline, self).__init__()
        self.pangu_alpha_embedding = PANGUALPHA_EmbeddingPipeLine(config).set_comm_fusion(1)
        self.pangu_alpha_embedding.stage = 0
        self.pangu_alpha_mask = PANGUALPHA_Mask(config)
        self.blocks = nn.CellList()
        dropout_recompute = False
        self.top_query_embedding = nn.Embedding(config.seq_length, config.embedding_size,
                                                embedding_table=TruncatedNormal(0.02))
        self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))
        self.top_query_embedding.expand.shard(((config.dp, 1),))
        for i in range(config.num_layers):
            if i == config.num_layers - 1:
                self.top_query_embedding.set_comm_fusion(2)
                self.top_query_embedding.stage = i * config.stage_num // config.num_layers
                per_block = QueryLayer(config).set_comm_fusion(2)
            else:
                per_block = Block(config, i + 1).set_comm_fusion(2)
            per_block.stage = i * config.stage_num // config.num_layers
            per_block.recompute()
            self.blocks.append(per_block)
            if not dropout_recompute:
                per_block.attention.dropout.dropout_gen_mask.recompute(False).add_prim_attr("_side_effect", True)
                per_block.attention.prob_dropout.dropout_gen_mask.recompute(False).add_prim_attr("_side_effect", True)
                per_block.output.dropout.dropout_gen_mask.recompute(False).add_prim_attr("_side_effect", True)

        if config.self_layernorm:
            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(mstype.float32)
        else:
            self.layernorm = nn.LayerNorm(
                (config.embedding_size,)).to_float(mstype.float32)
            self.layernorm.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        self.layernorm.set_comm_fusion(2)
        #self.layernorm.set_comm_fusion(3)
        self.layernorm.stage = config.stage_num - 1
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.dtype = config.compute_dtype
        self.num_layers = config.num_layers

    def construct(self, input_ids, input_mask, table, input_position, attention_mask, layer_past=None):
        """PANGUALPHA model"""
        if not self.use_past:
            layer_past = self.past

        hidden_states = self.pangu_alpha_embedding(input_ids, table, input_position)
        attention_mask = self.pangu_alpha_mask(input_mask, attention_mask)

        present_layer = ()
        for i in range(self.num_layers-1):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask, layer_past)
            present_layer = present_layer + (present,)
        top_query_hidden_states = self.top_query_embedding(input_position)
        hidden_states, present = self.blocks[self.num_layers-1](hidden_states, top_query_hidden_states,
                                                                attention_mask, layer_past)
        present_layer = present_layer + (present,)   
        output_state = self.layernorm(hidden_states)
        output_state = F.cast(output_state, self.dtype) 
        return output_state, present_layer


class PANGUALPHA_Model(nn.Cell):
    """
    The backbone of PANGUALPHA network
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs with datatype int32
        input_mask: the mask indicating whether each position is a valid input
        layer_past: the previous feature map
    Returns:
        output_state: Tensor, the output logit of backbone
        present_layer: Tensor, the current feature map
        embedding_table: Tensor, the embedding table for the vocabulary
    """

    def __init__(self, config):
        super(PANGUALPHA_Model, self).__init__()
        self.get_attention_mask = AttentionMask(config)
        self.word_embedding = EmbeddingLookup(config).set_comm_fusion(1)
        self.eod_reset = config.eod_reset
        if config.load_ckpt_path:
            # Loading the embedding table from the ckpt path:
            embedding_path = os.path.join(config.load_ckpt_path, 'position_embedding.npy')
            if os.path.exists(embedding_path):
                p_table = np.load(embedding_path)
                position_table_param = Tensor(p_table, mstype.float32)
            else:
                raise ValueError(f"{embedding_path} file not exits, please check whether position_embedding file exit.")
        else:
            position_table_param = TruncatedNormal(0.02)
            
        self.position_embedding = nn.Embedding(
            config.seq_length,
            config.embedding_size,
            embedding_table=position_table_param).set_comm_fusion(1)
        self.word_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.embedding_table.parallel_optimizer = False
        self.position_embedding.gather.shard(((1, 1), (config.dp,)))
        self.position_embedding.expand.shard(((config.dp, 1),))
        self.blocks = nn.CellList()
        fusion_group_num = 4
        fusion_group_size = config.num_layers // fusion_group_num
        fusion_group_size = max(fusion_group_size, 1)

        num_layers = config.num_layers - 1
        self.num_layers = num_layers

        for i in range(num_layers):
            per_block = Block(config, i + 1).set_comm_fusion(int(i / fusion_group_size) + 2)
            per_block.recompute()
            per_block.attention.dropout.dropout_gen_mask.recompute(False)
            per_block.attention.prob_dropout.dropout_gen_mask.recompute(False)
            per_block.output.dropout.dropout_gen_mask.recompute(False)
            per_block.attention.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            per_block.attention.prob_dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            per_block.output.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
            self.blocks.append(per_block)

        if config.self_layernorm:
            self.layernorm = LayerNorm((config.embedding_size,), config.dp).to_float(
                mstype.float32).set_comm_fusion(
                int((num_layers - 1) / fusion_group_size) + 2)
        else:
            self.layernorm = nn.LayerNorm((config.embedding_size,)).to_float(
                mstype.float32).set_comm_fusion(
                int((num_layers - 1) / fusion_group_size) + 2)
            self.layernorm.layer_norm.shard(((config.dp, 1, 1), (1,), (1,)))
        self.layernorm.gamma.parallel_optimizer = False
        self.layernorm.beta.parallel_optimizer = False
        self.use_past = config.use_past
        self.past = tuple([None] * config.num_layers)
        self.add = P.TensorAdd().shard(((config.dp, 1, 1), (config.dp, 1, 1)))
        self.expand_dims = P.ExpandDims().shard(((config.dp, 1, 1),))
        self.dtype = config.compute_dtype
        self.dropout = nn.Dropout(1 - config.dropout_rate)
        self.dropout.dropout_gen_mask.shard(((config.dp, 1, 1),))
        self.dropout.dropout_do_mask.shard(((config.dp, 1, 1),))

        if config.load_ckpt_path:
            # Loading the embedding table from the ckpt path:
            embedding_path = os.path.join(config.load_ckpt_path, 'top_query_embedding.npy')
            if os.path.exists(embedding_path):
                top_query_table = np.load(embedding_path)
                top_query_table_param = Tensor(top_query_table, mstype.float32)
            else:
                raise ValueError(f"{embedding_path} file not exits, please check whether top_query_embedding file exist.")
        else:
            top_query_table_param = TruncatedNormal(0.02)
            
        self.top_query_embedding = nn.Embedding(config.seq_length, config.embedding_size, \
                                                embedding_table=top_query_table_param).set_comm_fusion(
            int((config.num_layers - 1) / fusion_group_num) + 2)
        self.top_query_embedding.embedding_table.parallel_optimizer = False
        self.top_query_embedding.gather.shard(((1, 1), (config.dp,)))
        self.top_query_embedding.expand.shard(((config.dp, 1),))
        self.top_query_layer = QueryLayer(config)

        self.top_query_layer.recompute()

        self.top_query_layer.output.dropout.dropout_gen_mask.recompute(False)
        self.top_query_layer.attention.dropout.dropout_gen_mask.recompute(False)
        self.top_query_layer.attention.prob_dropout.dropout_gen_mask.recompute(False)

        self.top_query_layer.output.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
        self.top_query_layer.attention.dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)
        self.top_query_layer.attention.prob_dropout.dropout_gen_mask.add_prim_attr("_side_effect", True)

        self.top_query_layer.set_comm_fusion(int((config.num_layers - 1) / fusion_group_num) + 2)

    def construct(self, input_ids, input_mask, input_position=None, attention_mask=None, layer_past=None):
        """PanGu Alpha model"""
        if not self.use_past:
            layer_past = self.past

        input_embedding, embedding_table = self.word_embedding(input_ids)
        if not self.eod_reset:
            batch_size, seq_length = F.shape(input_ids)
            input_position = F.tuple_to_array(F.make_range(seq_length))
            input_position = P.Tile()(input_position, (batch_size, 1))
            attention_mask = self.get_attention_mask(input_mask)
            
        position_embedding = self.position_embedding(input_position)
        hidden_states = self.add(input_embedding, position_embedding)
        hidden_states = self.dropout(hidden_states)
        hidden_states = P.Cast()(hidden_states, mstype.float16)
        attention_mask = self.expand_dims(attention_mask, 1)

        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask, layer_past)
            present_layer = present_layer + (present,)

        output_state = self.layernorm(hidden_states)
        output_state = F.cast(output_state, self.dtype)

        top_query_hidden_states = self.top_query_embedding(input_position)
        output_state, present = self.top_query_layer(output_state, top_query_hidden_states,
                                                     attention_mask, layer_past)
        present_layer = present_layer + (present,)

        return output_state, present_layer, embedding_table


class PANGUALPHA_Head(nn.Cell):
    """
    Head for PANGUALPHA to get the logits of each token in the vocab
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        state: the output of the backbone
        embedding_table: the embedding table of the vocabulary
    Returns:
        logits: Tensor, the logits of the corresponding inputs
    """
    def __init__(self, config):
        super(PANGUALPHA_Head, self).__init__()
        if config.word_emb_dp:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (1, 1)))
        else:
            self.matmul = P.MatMul(transpose_b=True).shard(((config.dp, 1), (config.mp, 1)))
        self.embedding_size = config.embedding_size
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.compute_dtype
        self.cast = P.Cast()

    def construct(self, state, embedding_table):
        state = P.Reshape()(state, (-1, self.embedding_size))
        logits = self.matmul(state, self.cast(embedding_table, self.dtype))
        return logits


class PANGUALPHAPipeline(nn.Cell):
    """
    The PANGUALPHA network consisting of two parts the backbone and the head
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(PANGUALPHAPipeline, self).__init__()
        self.backbone = PANGUALPHA_ModelPipeline(config)
        self.head = PANGUALPHA_Head(config)
        self.head.stage = config.stage_num - 1
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.embedding_table = Parameter(initializer(
            Normal(0.02), [self.vocab_size, self.embedding_size]),
            name="embedding_table")

    def construct(self, input_ids, input_mask, input_position, attention_mask, past=None):
        output_states, _ = self.backbone(input_ids, input_mask, self.embedding_table, input_position, attention_mask, past)
        logits = self.head(output_states, self.embedding_table)
        return logits

class PANGUALPHA(nn.Cell):
    """
    The PANGUALPHA network consisting of two parts the backbone and the head
    Args:
        config(PANGUALPHAConfig): the config of network
    Inputs:
        input_ids: the tokenized inputs
        input_mask: the mask indicating whether each position is a valid input
        past: the previous feature map
    Returns:
        logits: Tensor: the logits of the corresponding inputs with shape (batch_size, seq_length, vocab_size)
    """
    def __init__(self, config):
        super(PANGUALPHA, self).__init__()
        self.backbone = PANGUALPHA_Model(config)
        self.head = PANGUALPHA_Head(config)

    def construct(self, input_ids, input_mask, input_position=None, attention_mask=None, past=None):
        output_states, _, embedding_table = self.backbone(
            input_ids, input_mask, input_position, attention_mask, past)
        logits = self.head(output_states, embedding_table)
        return logits

class CrossEntropyLoss(nn.Cell):
    """
    Calculate the cross entropy loss
    Args:
        config(PANGUALPHAConfig): the config of the network
    Inputs:
        logits: the output logits of the backbone
        label: the ground truth label of the sample
        input_mask: the mask indicating whether each position is a valid input
    Returns:
        loss: Tensor, the corrsponding cross entropy loss
    """
    def __init__(self, config):
        super(CrossEntropyLoss, self).__init__()
        self.mean = P.ReduceMean()
        self.sum = P.ReduceSum().shard(((config.dp, config.mp),))
        self.onehot = P.OneHot().shard(((config.dp, config.mp), (), ()))
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.vocab_size = config.vocab_size
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((config.dp, config.mp),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub().shard(((config.dp, config.mp), (config.dp, 1)))
        self.exp = P.Exp().shard(((config.dp, config.mp),))
        self.div = P.RealDiv().shard(((config.dp, config.mp), (config.dp, 1)))
        self.log = P.Log().shard(((config.dp, config.mp),))
        self.add = P.TensorAdd().shard(((config.dp, config.mp), ()))
        self.mul = P.Mul().shard(
            ((config.dp, config.mp), (config.dp, config.mp)))
        self.neg = P.Neg().shard(((config.dp, config.mp),))
        self.sum2 = P.ReduceSum().shard(((1,),))

        self.mul2 = P.Mul().shard(((1,), (1,)))
        self.add2 = P.TensorAdd()
        self.div2 = P.RealDiv()
        if config.stage_num > 1:
            self.div2.add_prim_attr("end", 0)

    def construct(self, logits, label, input_mask):
        logits = F.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))
        label = P.Reshape()(label, (-1,))
        one_hot_label = self.onehot(label, self.vocab_size, self.on_value,
                                    self.off_value)
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))

        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss


class MicroBatch(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.micro_slice = P.StridedSlice().shard(((1, 1),))
        self.micro_attention_slice = P.StridedSlice().shard(((1, 1),))
        self.shape = P.Shape()
        self.stage_num = config.micro_size
        self.seq_len =  config.seq_length
        self.slice_mask = P.StridedSlice().shard(((1, 1, 1),))

    def construct(self, x, i, input_position, attention_mask):
        input_shape = self.shape(x)
        micro_batch_begin = (i * input_shape[0] // self.stage_num, 0)
        micro_batch_end = ((i + 1) * input_shape[0] // self.stage_num, input_shape[1])
        micro_batch_stride = (1, 1)
        micro_input = self.micro_slice(x, micro_batch_begin, micro_batch_end, micro_batch_stride)
        micro_input_position_begin = (i * input_shape[0] // self.stage_num, 0)
        micro_input_position_end = ((i + 1) * input_shape[0] // self.stage_num, self.seq_len)
        micro_input_position = self.micro_attention_slice(input_position, micro_input_position_begin, micro_input_position_end, micro_batch_stride)
        micro_attention_mask_begin = (i * input_shape[0] // self.stage_num, 0, 0)
        micro_attention_mask_end = ((i + 1) * input_shape[0] // self.stage_num, self.seq_len, self.seq_len)
        micro_attention_mask_stride = (1, 1 ,1)
        micro_attention_mask = self.slice_mask(attention_mask, micro_attention_mask_begin, micro_attention_mask_end, micro_attention_mask_stride)
        return micro_input, micro_input_position, micro_attention_mask


class PANGUALPHAWithLossPipeline(nn.Cell):
    """
    PANGUALPHA training loss
    Args:
        network: backbone network of PANGUALPHA
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """
    def __init__(self, config, network, loss, eos_token=6):
        super(PANGUALPHAWithLossPipeline, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.micro_batch_step = config.micro_size
        self.micro_input = nn.CellList()
        self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))
        for i in range(self.micro_batch_step):
            micro = MicroBatch(config)
            micro.micro_slice.add_prim_attr("micro", i)
            micro.micro_slice.add_prim_attr("start", i)
            self.micro_input.append(micro)

    def construct(self, input_ids, input_position, attention_mask):
        #tokens = input_ids[:, :-1]
        ret = None
        for i in range(self.micro_batch_step):
            micro_input, micro_input_position, micro_attention_mask = self.micro_input[i](input_ids, i, input_position, attention_mask)
            tokens = self.slice(micro_input, (0, 0), (self.batch_size // self.micro_batch_step, -1), (1, 1))

            input_mask = F.cast(self.not_equal(tokens, self.eos_token), mstype.float32)
            logits = self.network(tokens, input_mask, micro_input_position, micro_attention_mask)
            labels = self.slice(micro_input, (0, 1), (self.batch_size // self.micro_batch_step,
                                                      self.len + 1), (1, 1))
            output = self.loss(logits, labels, input_mask)
            if ret is not None:
                ret = ret + output
            else:
                ret = output
        return ret


class PANGUALPHAWithLoss(nn.Cell):
    """
    PANGUALPHA training loss
    Args:
        network: backbone network of PANGUALPHA
        loss: loss function, e.g., crossentropy
        eos_token: the end_of_sentence token
    Inputs:
        input_ids: the tokenized inputs
        past: the previous feature map
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config, network, loss, eos_token=6):
        super(PANGUALPHAWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss
        self.eos_token = eos_token
        self.slice = P.StridedSlice().shard(((config.dp, 1),))
        self.not_equal = P.NotEqual().shard(((config.dp, 1), ()))
        self.batch_size = config.batch_size
        self.len = config.seq_length
        self.slice_mask = P.StridedSlice().shard(((config.dp, 1, 1),))

    def construct(self, input_ids, input_position=None, attention_mask=None):

        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))

        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        attention_mask = self.slice_mask(attention_mask, (0, 0, 0),
                                         (self.batch_size, self.len, self.len),
                                         (1, 1, 1))

        input_mask = F.cast(self.not_equal(tokens, self.eos_token),
                            mstype.float32)
        logits = self.network(tokens, input_mask, input_position, attention_mask)
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1),
                            (1, 1))
        output = self.loss(logits, labels, input_mask)
        return output


class EvalNet(nn.Cell):
    """
    PANGUALPHA evaluation net
    Args:
        backbone: backbone network of PANGUALPHA
        generate: enable generate mode
    Inputs:
        input_ids: the tokenized inpus
    Returns:
        outputs: Tensor, corresponding output for different tasks
    """
    def __init__(self, backbone, generate=False):
        super(EvalNet, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.argmax = P.ArgMaxWithValue()
        self.generate = generate
        self.topk = P.TopK(sorted=True).shard(((1, 1),))
        self.log_softmax = P.Softmax(axis=-1)

    def construct(self, input_ids):
        """evaluation net"""
        input_mask = F.cast(F.not_equal(input_ids, 6), mstype.float32)
        logits = self.backbone(input_ids, input_mask)
        value, index = self.topk(logits, 5)
        probs = self.log_softmax(value)
        return probs, index
