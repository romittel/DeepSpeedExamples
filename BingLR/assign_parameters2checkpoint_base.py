import torch
import os
indir = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints/50/'
indir0 = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints/100000/'
outdir = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints/50_binglr/'
num_of_layers = 24
num_of_gpus = 8
num_of_attention_heads = 16
hp = 1024 // num_of_gpus
h = 1024
d_binglr = torch.load('/relevance2-nfs/romittel/binglr_pretrained_model/pytorch_model.bin')

checkpoints = ['mp_rank_00_model_states.pt', 'mp_rank_01_model_states.pt', 'mp_rank_02_model_states.pt', 'mp_rank_03_model_states.pt', 'mp_rank_04_model_states.pt', 'mp_rank_05_model_states.pt', 'mp_rank_06_model_states.pt', 'mp_rank_07_model_states.pt']

for j in range(len(checkpoints)):
    d_moe = torch.load(os.path.join(indir, checkpoints[j]))
    d_moe0 = torch.load(os.path.join(indir0, checkpoints[j]))
    for key in d_moe['module'].keys():
        d_moe['module'][key] = d_moe0['module'][key]

    
    torch.save(d_moe, os.path.join(outdir, checkpoints[j]))

#['word_embeddings.weight', 'position_embeddings.weight', 'token_type_embeddings.weight', 'input_layernorm.weight', 'input_layernorm.bias', 
#'transformer.layers.0.attention.query_key_value.weight', 'transformer.layers.0.attention.query_key_value.bias', 
#'transformer.layers.0.self_output.dense.weight', 'transformer.layers.0.self_output.dense.bias', 
#'transformer.layers.0.self_output.layernorm.weight', 'transformer.layers.0.self_output.layernorm.bias', 

#BingLR:
#bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight', 
#'bert.embeddings.token_type_embeddings.weight',  'bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 


#'bert.encoder.layer.0.attention.self.query.weight', 
#'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.key.weight', 'bert.encoder.layer.0.attention.self.key.bias',
# 'bert.encoder.layer.0.attention.self.value.weight', 'bert.encoder.layer.0.attention.self.value.bias', 
#'bert.encoder.layer.0.attention.output.dense.weight', 'bert.encoder.layer.0.attention.output.dense.bias', 
