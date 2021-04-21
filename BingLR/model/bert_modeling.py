# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
import torch.nn.functional as F
import mpu
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from mpu.layers import ColumnParallelLinear, RowParallelLinear
import torch.nn.init as init
def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class BertMixtureModel(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 num_experts=1,
                 type_vocab_size=2):

        super(BertMixtureModel, self).__init__()

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)
        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Token Type Enbeddings.
        self.token_type_embeddings = torch.nn.Embedding(type_vocab_size, hidden_size)

        # Initialize the token type embeddings.
        init_method(self.token_type_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        self.hrs_head = RowParallelLinear(
            hidden_size,
            1,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.hrs_head.weight)

        self.click_head = RowParallelLinear(
            hidden_size,
            1,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.click_head.weight)

        self.lpsat_head = RowParallelLinear(
            hidden_size,
            5,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.lpsat_head.weight)

        self.qc_head = RowParallelLinear(
            hidden_size,
            5,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.qc_head.weight)

        self.eff_head = RowParallelLinear(
            hidden_size,
            5,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.eff_head.weight)

        self.local_head = RowParallelLinear(
            hidden_size,
            5,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.local_head.weight)

        self.fresh_head = RowParallelLinear(
            hidden_size,
            5,
            input_is_parallel=True,
            init_method=init.xavier_normal_)

        init_method(self.fresh_head.weight)

        # Transformer
        self.transformer = mpu.BertParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       num_experts=num_experts)

        self.dropout = torch.nn.Dropout(output_dropout_prob)

        self.dense_hrs0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)


        self.dense_click0 = ColumnParallelLinear(hidden_size,
                                       hidden_size,
                                       gather_output=False,
                                       init_method=init.xavier_normal_)
        
        
        self.dense_lpsat0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)

        self.dense_qc0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)

        self.dense_eff0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)

        self.dense_local0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)

        self.dense_fresh0 = ColumnParallelLinear(hidden_size, hidden_size,
                                                  gather_output=False,
                                                  init_method=init.xavier_normal_)

    def forward(self, input_ids, position_ids, attention_mask, token_type_ids):

        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        embeddings = self.input_layernorm(embeddings)
        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        transformer_output, *moe_losses = self.transformer(embeddings, attention_mask)
        
        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            transformer_output)
        logits_parallel = F.linear(transformer_output_parallel,
                                   self.word_embeddings.weight)
        
        pooled_output = torch.squeeze(transformer_output_parallel[:,0,:])
        ##############
        #hrs_scores = self.hrs_head(pooled_output)
        #click_scores = self.click_head(pooled_output)
        #############
        hrs_head0 = self.dense_hrs0(pooled_output)
        hrs_scores = self.hrs_head(torch.tanh(hrs_head0))
 
        click_head0 = self.dense_click0(pooled_output)
        click_scores = self.click_head(torch.tanh(click_head0))

        lpsat_head0 = self.dense_hrs0(pooled_output)
        lpsat_scores = self.hrs_head(torch.tanh(lpsat_head0))

        qc_head0 = self.dense_hrs0(pooled_output)
        qc_scores = self.hrs_head(torch.tanh(qc_head0))

        eff_head0 = self.dense_hrs0(pooled_output)
        eff_scores = self.hrs_head(torch.tanh(eff_head0))

        local_head0 = self.dense_hrs0(pooled_output)
        local_scores = self.hrs_head(torch.tanh(local_head0))

        fresh_head0 = self.dense_hrs0(pooled_output)
        fresh_scores = self.hrs_head(torch.tanh(fresh_head0))
        #############
        if self.parallel_output:
            return (logits_parallel, hrs_scores, click_scores, *moe_losses)
        
        return (mpu.gather_from_model_parallel_region(logits_parallel), hrs_scores, click_scores, *moe_losses) 

def bert_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
