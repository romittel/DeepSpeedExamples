import torch
import os
from collections import OrderedDict
indir_mlm = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/backup_models/1580000/'
indir_binglr = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints_binglr/50/'
outdir = '/relevance2-nfs/romittel/DeepSpeedExamples-amawa-moe/Megatron-LM-base-iterator/checkpoints_binglr/50/'
checkpoints = ['mp_rank_00_model_states.pt', 'mp_rank_01_model_states.pt', 'mp_rank_02_model_states.pt', 'mp_rank_03_model_states.pt', 'mp_rank_04_model_states.pt', 'mp_rank_05_model_states.pt', 'mp_rank_06_model_states.pt', 'mp_rank_07_model_states.pt']


for j in range(len(checkpoints)):
    d_mlm = torch.load(os.path.join(indir_mlm, checkpoints[j]), map_location=torch.device('cpu'))
    d_binglr = torch.load(os.path.join(indir_binglr, checkpoints[j]), map_location=torch.device('cpu'))
    new_binglr_module = OrderedDict()
    for k,v in d_binglr['module'].items():
        if k in d_mlm['module'].keys():
            new_binglr_module[k] = d_mlm['module'][k]
        else:
            new_binglr_module[k] = v
    d_binglr['module'] = new_binglr_module
    d_binglr['optimizer']['cur_scale'] = d_mlm['optimizer']['cur_scale']
    torch.save(d_binglr, os.path.join(outdir, checkpoints[j]))


