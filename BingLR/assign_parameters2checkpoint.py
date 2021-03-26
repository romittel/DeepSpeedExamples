import torch
import os
indir = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints/50/'
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

    print(d_moe['module']['word_embeddings.weight'].size())
    print(d_binglr['bert.embeddings.word_embeddings.weight'].size())
for j in range(len(checkpoints)):
    d_moe = torch.load(os.path.join(indir, checkpoints[j]))
    emb_per_gpu = d_binglr['bert.embeddings.word_embeddings.weight'].size()[0] // num_of_gpus
    assert(d_moe['module']['word_embeddings.weight'][:emb_per_gpu,:].size()[1] == d_binglr['bert.embeddings.word_embeddings.weight'].size()[1])
    d_moe['module']['word_embeddings.weight'][:emb_per_gpu,:] = d_binglr['bert.embeddings.word_embeddings.weight'][j * emb_per_gpu : (j + 1) * emb_per_gpu,:]

    #print(d_moe['module']['position_embeddings.weight'].size())
    #print(d_binglr['bert.embeddings.position_embeddings.weight'].size())
    assert(d_moe['module']['position_embeddings.weight'].size() == d_binglr['bert.embeddings.position_embeddings.weight'].size())
    d_moe['module']['position_embeddings.weight'] = d_binglr['bert.embeddings.position_embeddings.weight']

    #print(d_moe['module']['token_type_embeddings.weight'].size())
    #print(d_binglr['bert.embeddings.token_type_embeddings.weight'].size())
    #assert(d_moe['module']['token_type_embeddings.weight'].size() == d_binglr['bert.embeddings.token_type_embeddings.weight'].size())
    #d_moe['module']['token_type_embeddings.weight'] = d_binglr['bert.embeddings.token_type_embeddings.weight']

    assert(d_moe['module']['input_layernorm.weight'].size() == d_binglr['bert.embeddings.LayerNorm.weight'].size())
    d_moe['module']['input_layernorm.weight'] = d_binglr['bert.embeddings.LayerNorm.weight']

    assert(d_moe['module']['input_layernorm.bias'].size() == d_binglr['bert.embeddings.LayerNorm.bias'].size())
    d_moe['module']['input_layernorm.bias'] = d_binglr['bert.embeddings.LayerNorm.bias']

    for i in range(num_of_layers):
        query_weight = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.query.weight'] 
        query_bias = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.query.bias'] 
        key_weight = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.key.weight'] 
        key_bias = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.key.bias'] 
        value_weight = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.value.weight'] 
        value_bias = d_binglr['bert.encoder.layer.' + str(i) + '.attention.self.value.bias']
        query_key_value_weight = d_moe['module']['transformer.layers.' + str(i) + '.attention.query_key_value.weight']
        query_key_value_bias = d_moe['module']['transformer.layers.' + str(i) + '.attention.query_key_value.bias']

        query_key_value_weight[: hp,:] =  query_weight[j * hp : (j+1) * hp,:]
        query_key_value_weight[hp:  hp * 2,:] =  key_weight[j * hp : (j+1) * hp,:]
        query_key_value_weight[2 * hp:,:] =  value_weight[j * hp : (j+1) * hp,:]
        query_key_value_bias[:hp] = query_bias[j * hp : (j+1) * hp]
        query_key_value_bias[hp:  hp * 2] = key_bias[j * hp : (j+1) * hp]
        query_key_value_bias[2 * hp:] =  value_bias[j * hp : (j+1) * hp]
        d_moe['module']['transformer.layers.' + str(i) + '.attention.query_key_value.weight'] = query_key_value_weight
        d_moe['module']['transformer.layers.' + str(i) + '.attention.query_key_value.bias'] = query_key_value_bias

        print(d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.weight'].size())
        print(d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.weight'].size())
        assert(d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.weight'].size() == d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.weight'][:,j * hp : (j+1) * hp].size())
        d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.weight'] = d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.weight'][:,j * hp : (j+1) * hp]

        print(d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.bias'].size())
        print(d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.bias'].size())
        assert(d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.bias'].size() == d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.bias'].size())
        d_moe['module']['transformer.layers.' + str(i) + '.self_output.dense.bias'] = d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.dense.bias']
        
        print(d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.weight'].size())
        print(d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.weight'].size())
        assert(d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.weight'].size() == d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.weight'].size())
        d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.weight'] = d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.weight']


        print(d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.bias'].size())
        print(d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.bias'].size())
        assert(d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.bias'].size() == d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.bias'].size())
        d_moe['module']['transformer.layers.' + str(i) + '.self_output.layernorm.bias'] = d_binglr['bert.encoder.layer.' + str(i) + '.attention.output.LayerNorm.bias']

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
