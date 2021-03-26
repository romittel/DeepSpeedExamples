#! /bin/bash

# Change for multinode config
MP_SIZE=8

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero2_config.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 25 \
       --seq-length 128 \
       --max-position-embeddings 512 \
       --train-iters 10000000 \
       --resume-dataloader \
       --tokenizer-type BertSentencePieceTokenizer \
       --split 1 \
       --max-preds-per-seq 20 \
       --mask-lm-prob 0.15 \
       --distributed-backend nccl \
       --lr 0.000015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .000001 \
       --init-method-std 0.001 \
       --fp16 \
       --num-workers 1 \
       --num-experts 32 \
       --expert-parallel-size 1 \
       --log-interval 10 \
       --train-data /relevance2-nfs/local/users/xiaolhu/V3_attempt1 \
       --valid-data /relevance2-nfs/romittel/binglr_validation_data.json \
       --save /relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints_mlm \
       --tokenizer-path /relevance2-nfs/romittel/binglr_pretrained_model/ \
       --text-key docs \
       --label-key task_id \
       --loose-json \
       --vocab-size 250368 \
       --save-interval 20000 \
       --eval-interval 5000 \
       --eval-iters 100 \
       --num-urls 4 \
       --train-file-lens-path /relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/file_lens.tsv \
       --load /relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints_mlm/
"

# Disable activation checkpointing

#     --checkpoint-activations \
#       --deepspeed-activation-checkpointing \

gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2.8.3 deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_only_bert_mixture.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
