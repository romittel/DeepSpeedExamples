#!/bin/bash

CHECKPOINT_PATH=/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/backup_models
MPSIZE=8
NLAYERS=24
NHIDDEN=1024
NATT=16
MAXSEQLEN=128

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

python generate_moe_predictions.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
	   --num-attention-heads $NATT \
       --load $CHECKPOINT_PATH \
	   --valid-data /relevance2-nfs/romittel/validation_data_binglr_full_moe.tsv \
	   --output-path /relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/output_dir/moe_predictions.tsv \
	   --max-position-embeddings 512 \
       --tokenizer-type BertSentencePieceTokenizer \
	   --text-key docs \
       --label-key task_id \
	   --seq-length $MAXSEQLEN \
	   --num-experts 32 \
	   --batch-size 1\
       --fp16
