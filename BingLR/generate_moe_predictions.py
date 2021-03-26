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

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from arguments import get_args
from utils import Timers
from data_utils.datasets import bert_sentencepair_dataset
import mpu
#from pretrain_gpt2 import initialize_distributed
#from pretrain_gpt2 import set_random_seed
#from pretrain_gpt2 import get_train_val_test_data
#from pretrain_gpt2 import get_masks_and_position_ids
from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data
from data_utils.tokenization import make_tokenizer

from fp16 import FP16_Module
from model import BertMixtureModel
from model import DistributedDataParallel as DDP
from utils import print_rank_0

def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed

def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)

def get_model(args):
    """Build the model."""

    print_rank_0('building Bert MoE model ...')
    model = BertMixtureModel(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      layernorm_epsilon=args.layernorm_epsilon,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False,
                      num_experts=args.num_experts,
                      type_vocab_size=2)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = DDP(model)

    return model

def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        _ = load_checkpoint(
            model, None, None, args)

    return model


#def get_batch(context_tokens, device, args):
#    tokens = context_tokens
#    tokens = tokens.view(args.batch_size, -1).contiguous()
#    tokens = tokens.to(device)
#
#    # Get the masks and postition ids.
#    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
#        tokens,
#        args.eod_token,
#        args.reset_position_ids,
#        args.reset_attention_mask)
#
#    return tokens, attention_mask, position_ids

#def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
#    # This function has been mostly taken from huggingface conversational ai code at
#    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
#
#    if top_k > 0:
#        # Remove all tokens with a probability less than the last token of the top-k
#        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
#        logits[indices_to_remove] = filter_value
#        
#    if top_p > 0.0:
#        #convert to 1D
#        logits=logits.view(logits.size()[1]).contiguous()
#        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
#        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#
#        # Remove tokens with cumulative probability above the threshold
#        sorted_indices_to_remove = cumulative_probs > top_p
#        # Shift the indices to the right to keep also the first token above the threshold
#        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#        sorted_indices_to_remove[..., 0] = 0
#        indices_to_remove = sorted_indices[sorted_indices_to_remove]
#        logits[indices_to_remove] = filter_value
#        #going back to 2D
#        logits=logits.view(1, -1).contiguous()
#	
#    return logits


def generate_samples(model, tokenizer, args, device):
    
    context_count=0
    keys = ['text', 'types', 'mask', 'mask_labels', 'pad_mask']
    datatype = torch.int64
    keys2 = ['clickscores', 'hrsscores']
    datatype2 = torch.float64
    model.eval()
    fout = open(args.output_path, 'w', encoding='utf-8')
    with torch.no_grad():
        data_ietrator = binglr_iterator_dataset([args.valid_data], run_once=True, max_seq_len=args.seq_length, mask_lm_prob=0.15, max_preds_per_seq=20, tokenizer=tokenizer, train=False)
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs=0
            if mpu.get_model_parallel_rank() == 0:
                data = next(data_ietrator)
                 
                if sample is None:
                    terminate_runs = 1
                else: 
                    # Unpack.
                    tokens = data['text']
                    types = data['types']
                    loss_mask = data['mask']
                    lm_labels = data['mask_labels']
                    padding_mask = data['pad_mask']
                    clickscores = data['clickscores']
                    hrsscores = data['hrsscores']
                    sample_id = data['sample_id']
                    # Get the masks and postition ids.
                   
            else:
                tokens = np.array([0] * seq_length)
                types = np.array([0] * seq_length)
                loss_mask = np.array([0] * seq_length)
                lm_labels = np.array([0] * seq_length)
                padding_mask = np.array([0] * seq_length)
                clickscores = np.array([0.0])
                hrsscores = np.array([0.0])
                sample_id = np.array([0])
            
            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            tokens_tensor = torch.cuda.LongTensor(tokens).view(1,-1)
            types_tensor = torch.cuda.LongTensor(types).view(1,-1)
            loss_mask_tensor = torch.cuda.LongTensor(loss_mask).view(1,-1)
            lm_labels_tensor = torch.cuda.LongTensor(lm_labels).view(1,-1)
            padding_mask_tensor = torch.cuda.LongTensor(padding_mask_mask).view(1,-1)
            clickscores_tensor = torch.cuda.FloatTensor(clickscores)
            hrsscores_tensor = torch.cuda.FloatTensor(hrsscores)
            batch_size, seq_length = tokens.size()
            attention_mask_tensor = (torch.ones_like(padding_mask, device=padding_mask.device) - padding_mask).view(batch_size, 1, seq_length, 1) * (torch.ones_like(padding_mask, device=padding_mask.device) - padding_mask).view(batch_size, 1, 1, seq_length)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
            position_ids_tensor = position_ids.unsqueeze(0).expand_as(tokens)
            
            torch.distributed.broadcast(tokens_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(types_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(clickscores_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(hrsscores_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(attention_mask_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(position_ids_mask_tensor, mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

            start_time = time.time()

            counter = 0
            _, hrs_scores, _, _ = model(tokens.contiguous(), position_ids.contiguous(), attention_mask.contiguous(), types.contiguous())

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                fout('\t'.join([sample_id, hrs_scores.detach().clone().cpu().numpy()]) + '\n')

            torch.distributed.barrier(group=mpu.get_model_parallel_group())

#def prepare_tokenizer(args):
#
#    tokenizer_args = {
#        'tokenizer_type': args.tokenizer_type,
#        'corpus': None,
#        'model_path': args.tokenizer_path,
#        'vocab_size': args.vocab_size,
#        'model_type': args.tokenizer_model_type,
#        'cache_dir': args.cache_dir}
#    tokenizer = make_tokenizer(**tokenizer_args)
#
#    args.tokenizer_num_tokens = tokenizer.num_tokens
#    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
#    args.eod_token = tokenizer.get_command('eos').Id
#
#    after = tokenizer.num_tokens
#    while after % mpu.get_model_parallel_world_size() != 0:
#        after += 1
#
#    args.vocab_size = after
#    print("prepare tokenizer done", flush=True)
#
#    return tokenizer

def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = make_tokenizer(args.tokenizer_type, None, args.tokenizer_path, args.vocab_size, args.tokenizer_model_type, 
                                    pad_token=0, character_converage=1.0)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    #setting default batch size to 1
    args.batch_size = 1

    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

if __name__ == "__main__":
    main()



