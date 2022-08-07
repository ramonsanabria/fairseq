from prettytable import PrettyTable
import fairseq
import sys

def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model[0].parameters() if p.requires_grad)
    print(model[0])

    return pytorch_total_params

#ckpt_path="/disk/scratch1/ramons/data/hubert_models/hubert_xtralarge_ll60k.pt"
ckpt_path="/disk/scratch1/ramons/data/hubert_models/hubert_large_ll60k.pt"
#ckpt_path="/disk/scratch1/ramons/data/w2v2_models/wav2vec_vox_new.pt"
#ckpt_path="/disk/scratch1/ramons/data/w2v2_models/xlsr_53_56k.pt"
(model,cfg,task,) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
count_parameters(model)
