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

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

from datetime import datetime
import os
import random
import math
import numpy as np
import torch

import deepspeed
import deepspeed.utils.groups as groups
from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import BertMixtureModel
from model import bert_get_params_for_weight_decay_optimization
from model import PairwiseHRSLoss, PairwiseClickLoss
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_params_min_max_norm
from utils import print_rank_0
import torch.distributed as dist

from gpt2_data_loader import make_gpt2_dataloaders

XDCG_DISCOUNT = 0.6

def get_model(args):
    """Build the model."""
    
    print_rank_0('building Bert model ...')
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
                      parallel_output=True,
                      num_experts=args.num_experts,
                      type_vocab_size=2)
    
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    #To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = bert_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )
        print(model)
    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


#def get_batch(data_iterator, args, timers):
#    ''' get_batch subdivides the source data into chunks of
#    length args.seq_length. If source is equal to the example
#    output of the data loading example, with a seq_length limit
#    of 2, we'd get the following two Variables for i = 0:
#    ┌ a g m s ┐ ┌ b h n t ┐
#    └ b h n t ┘ └ c i o u ┘
#    Note that despite the name of the function, the subdivison of data is not
#    done along the batch dimension (i.e. dimension 1), since that was handled
#    by the data loader. The chunks are along dimension 0, corresponding
#    to the seq_len dimension in the LSTM. A Variable representing an appropriate
#    shard reset mask of the same dimensions is also returned.
#    '''
#    # Items and their type.
#    keys = ['text']
#    datatype = torch.int64
#
#    # Broadcast data.
#    timers('data loader').start()
#    if data_iterator is not None:
#        data = next(data_iterator)
#    else:
#        data = None
#    timers('data loader').stop()
#    data_b = mpu.broadcast_data(keys, data, datatype)
#
#    # Unpack.
#    tokens_ = data_b['text'].long()
#    labels = tokens_[:, 1:].contiguous()
#    tokens = tokens_[:, :-1].contiguous()
#
#    # Get the masks and postition ids.
#    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
#        tokens,
#        args.eod_token,
#        args.reset_position_ids,
#        args.reset_attention_mask)
#    # Convert
#    if args.fp16:
#        attention_mask = attention_mask.half()
#
#    token_type_ids = torch.zeros_like(tokens, dtype=tokens.dtype, device=tokens.device)
#    return tokens, labels, loss_mask, attention_mask, position_ids, token_type_ids

def get_batch(data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'types', 'mask', 'mask_labels', 'pad_mask']
    datatype = torch.int64
    keys2 = ['clickscores', 'hrsscores']
    datatype2 = torch.float64
    # Broadcast data.
    timers('data loader').start()
    #if torch.distributed.get_rank() == 0:
    #    print("CCCCCCCCCCCCCCCCCCCCCCCCCC")
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    #if torch.distributed.get_rank() == 0:
    #    print("DDDDDDDDDDDDDDDDDDDDDDDDD")
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    data_b2 = mpu.broadcast_data(keys2, data, datatype2)

    # Unpack.

    tokens = data_b['text'].long()
    batch_size, num_urls, seq_length = tokens.size()
    tokens = data_b['text'].view(-1, seq_length).long()
    types = data_b['types'].view(-1, seq_length).long()
    #if torch.distributed.get_rank() == 0:
    #    print("tokens= ", tokens[0:4,:].detach().cpu().numpy())
    loss_mask = data_b['mask'].view(-1, seq_length).float()
    lm_labels = data_b['mask_labels'].view(-1, seq_length).long()
    #if torch.distributed.get_rank() == 0:
    #    print("lm_labels= ", lm_labels[0:4,:].detach().cpu().numpy())
    padding_mask = data_b['pad_mask'].view(-1, seq_length).float()
    clickscores = data_b2['clickscores'].view(batch_size, num_urls).float()
    hrsscores = data_b2['hrsscores'].view(batch_size, num_urls).float()
    # Get the masks and postition ids.
    batch_size, seq_length = tokens.size()
    attention_mask = (torch.ones_like(padding_mask, device=padding_mask.device) - padding_mask).view(batch_size, 1, seq_length, 1) * (torch.ones_like(padding_mask, device=padding_mask.device) - padding_mask).view(batch_size, 1, 1, seq_length)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()
    
    #types = torch.zeros_like(tokens, dtype=torch.long, device=types.device)
    return tokens, types, loss_mask, lm_labels, padding_mask, attention_mask, position_ids, clickscores, hrsscores

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    #if torch.distributed.get_rank() == 0:
    #    print("AAAAAAAAAAAAAAAAAA")
    tokens, types, loss_mask, lm_labels, padding_mask, attention_mask, position_ids, clicklabels, hrslabels = get_batch(
        data_iterator, args, timers)
    #if torch.distributed.get_rank() == 0:
    #    print("BBBBBBBBBBBBBBBBBB")
    timers('batch generator').stop()

    # Forward model.
    output, hrs_scores, click_scores, *other_losses = model(tokens, position_ids, attention_mask, types)
    #pooled_output = torch.squeeze(output[:,0,:])
    
    losses = mpu.vocab_parallel_cross_entropy(
        output.contiguous().float(), lm_labels.contiguous())
    loss_mask = loss_mask.contiguous()
    lm_loss = torch.sum(
        losses.view(-1) * loss_mask.view(-1).float()) / loss_mask.sum()
    loss = lm_loss

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)      

    hrs_loss = PairwiseHRSLoss(hrs_scores.contiguous(), hrslabels.contiguous(), 4)
    click_loss = PairwiseClickLoss(click_scores.contiguous(), clicklabels.contiguous(), 4)
    
    #print(f"Moe Losses: {moe_losses}, actual loss {loss}")
    moe_loss = sum(moe_losses)
    #if torch.distributed.get_rank() == 0:
    #    print(f"Moe Loss {moe_loss}")
    loss = loss + moe_loss * 0.1 + hrs_loss + click_loss
    
    return loss,  hrs_scores, hrslabels, lm_loss, hrs_loss, click_loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""

    # Total loss.
    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced

def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", torch.cuda.memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Max Memory Allocated ", torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Cache Allocated ", torch.cuda.memory_cached()/(1024*1024*1024), "GigaBytes")
        print("Max cache Allocated ", torch.cuda.max_memory_cached()/(1024*1024*1024), "GigaBytes")
        print(" ")
        #input("Press Any Key To Continue ..")

def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    # Forward model for one step.
    timers('forward').start()
    loss, _, _, lm_loss, hrs_loss, click_loss = forward_step(data_iterator, model, args, timers)
    timers('forward').stop()

    #print_rank_0("loss is {}".format(lm_loss))

    # Calculate gradients, reduce across processes, and clip.
    timers('backward').start()
    loss_reduced = backward_step(optimizer, model, loss, args, timers)
    timers('backward').stop()

    # Update parameters.
    skipped_iter = 0
    timers('optimizer').start()
    if args.deepspeed:
        model.step()
    else:
        optimizer.step()

        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1
    timers('optimizer').stop()

    return loss_reduced, skipped_iter, lm_loss, hrs_loss, click_loss


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_lm_loss0 = 0.0
    total_hrs_loss = 0.0
    total_click_loss = 0.0
    # Iterations.
    iteration = args.iteration
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    val_xdcg_at_3 = None
    while iteration < args.train_iters:

        lm_loss, skipped_iter, lm_loss0, hrs_loss, click_loss = train_step(train_data_iterator,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers)
        skipped_iters += skipped_iter
        iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        total_lm_loss0 += lm_loss0.view(1).data.detach().float()
        total_hrs_loss += hrs_loss.view(1).data.detach().float()
        total_click_loss += click_loss.view(1).data.detach().float()
        # Logging.
        if iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            avg_lm_loss0 = total_lm_loss0.item() / args.log_interval
            avg_hrs_loss = total_hrs_loss.item() / args.log_interval
            avg_click_loss = total_click_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3E} |'.format(learning_rate)
            log_string += ' total loss {:.6E} |'.format(avg_lm_loss)
            log_string += ' lm loss {:.6E} |'.format(avg_lm_loss0)
            log_string += ' hrs loss {:.6E} |'.format(avg_hrs_loss)
            log_string += ' click loss {:.6E} |'.format(avg_click_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
            print_rank_0(log_string)
            total_lm_loss = 0.0
            total_lm_loss0 = 0.0
            total_hrs_loss = 0.0
            total_click_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
            if USE_TORCH_DDP:
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        #if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
        #    prefix = 'iteration {}'.format(iteration)
        #    
        #    val_xdcg_at_3_new = evaluate_and_print_results(
        #        prefix, val_data_iterator, model, args, timers, False)
        #    if val_xdcg_at_3 is None:
        #        val_xdcg_at_3 = val_xdcg_at_3_new
        #        save_checkpoint(200, model, optimizer, lr_scheduler, args)
        #    elif val_xdcg_at_3_new > val_xdcg_at_3:
        #        val_xdcg_at_3 = val_xdcg_at_3_new
        #        save_checkpoint(200, model, optimizer, lr_scheduler, args)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def compute_xdcg(docs_label_score, depth=1):
    """Compute and return XDCG given a list ranked documents."""
    if depth <= 0:
        raise Exception("Invalid depth for xdcg calculation.")

    xdcg = np.zeros(depth)
    num_docs = len(docs_label_score)

    for i in range(depth):
        # current gain
        if i < num_docs:
            xdcg_label = float(docs_label_score[i][0]) * 25
            xdcg[i] = xdcg_label * math.pow(XDCG_DISCOUNT, i)
        # add previous gain
        if i > 0:
            xdcg[i] += xdcg[i - 1]

    return xdcg

def evaluate(data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_xdcg_at_3 = 0

    with torch.no_grad():
        iteration = 0
        for _ in range(args.eval_iters):
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            _, hrs_scores, hrslabels, _, _, _ = forward_step(data_iterator, model, args, timers)
            
            hrs_scores_value = hrs_scores.view(-1, 4).detach().cpu().numpy()
            hrslabels_value = hrslabels.view(-1, 4).detach().cpu().numpy()

            xdcg_at_3_sum = 0
            for i in range(hrslabels_value.shape[0]):
                label_score_list = [(label, score) for label, score in zip(hrslabels_value[i], hrs_scores_value[i])]
                label_score_list = sorted(label_score_list, key=lambda x: x[1])
                xdcg = compute_xdcg(label_score_list, depth=3)
                xdcg_at_3_sum += xdcg[2]
            
            xdcg_at_3 = xdcg_at_3_sum / float(hrslabels_value.shape[0])
            total_xdcg_at_3 += xdcg_at_3

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

    # Move model back to the train mode.
    model.train()

    total_xdcg_at_3 /= float(iteration)
    return total_xdcg_at_3


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    xdcg_at_3 = evaluate(data_iterator, model, args, timers, verbose)
    print_rank_0('-' * 100)
    string = ' validation at {} | '.format(prefix)
    string += 'xDCG@3: {:.6E} | '.format(xdcg_at_3)
    length = len(string) + 1
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)

    return xdcg_at_3

'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
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
    groups.initialize(ep_size=args.expert_parallel_size, mpu=mpu)
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


def get_train_val_test_data(args):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)

    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='BERT', transpose=False)
        (train_data, val_data, test_data), tokenizer = data_config.apply(args)
        before = tokenizer.num_tokens
        after = before
        multiple = args.make_vocab_size_divisible_by * \
                   mpu.get_model_parallel_world_size()
        while (after % multiple) != 0:
            after += 1
        print_rank_0('> padded vocab (size: {}) with {} dummy '
                     'tokens (new size: {})'.format(
                         before, after - before, after))
        # Need to broadcast num_tokens and num_type_tokens.
        token_counts = torch.cuda.LongTensor([after,
                                              tokenizer.num_type_tokens,
                                              int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        token_counts = torch.cuda.LongTensor([0, 0, 0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(token_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_tokens = token_counts[0].item()
    num_type_tokens = token_counts[1].item()
    args.do_train = token_counts[2].item()
    args.do_valid = token_counts[3].item()
    args.do_test = token_counts[4].item()

    return train_data, val_data, test_data, num_tokens, num_type_tokens




def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        print_args(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    train_data, val_data, test_data, args.vocab_size, \
        args.eod_token = get_train_val_test_data(args)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    #model.optimizer.dynamic_loss_scale=True
    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % \
                                                  len(train_data)
        if val_data is not None:
            start_iter_val = (args.train_iters // args.save_interval) * \
                             args.eval_interval
            val_data.batch_sampler.start_iter = start_iter_val % \
                                                len(val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

     
    
    #TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            iteration, skipped = train(model, optimizer,
                                       lr_scheduler,
                                       train_data_iterator,
                                       val_data_iterator,
                                       timers, args)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, timers, True)


if __name__ == "__main__":
    main()
