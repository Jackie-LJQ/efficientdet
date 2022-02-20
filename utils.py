from effdet.object_detection import Det_Dual_BN
import torch.nn as nn
import torch

def get_clip_parameters(model, exclude_head=False):
    if exclude_head:
        # FIXME this a bit of a quick and dirty hack to skip classifier head params
        return [p for n, p in model.named_parameters() if 'predict' not in n]
    else:
        return model.parameters()

def set_advState(model, advState):
    for name, submodel in model.named_modules():
        if isinstance(submodel, Det_Dual_BN):
            submodel.advState = advState

def convert_dual_bn(model):
    model_output = model
    if isinstance(model, nn.BatchNorm2d):
        model_output = Det_Dual_BN(model.num_features, model.eps, 
                    model.momentum, model.affine, model.track_running_stats)
        if model.affine:
            with torch.no_grad():
                model_output.bn_clean.weight = model.weight
                model_output.bn_clean.bias = model.bias
                model_output.bn_adv.weight = model.weight
                model_output.bn_adv.bias = model.bias
                
        model_output.bn_clean.running_mean = model.running_mean
        model_output.bn_clean.running_var = model.running_var
        model_output.bn_clean.num_batches_tracked = model.num_batches_tracked
        model_output.bn_adv.running_mean = model.running_mean
        model_output.bn_adv.running_var = model.running_var
        model_output.bn_adv.num_batches_tracked = model.num_batches_tracked
    
    for name, child in model.named_children():
        model_output.add_module(name, convert_dual_bn(child))
    
    del model
    return model_output

def convert_model(state_dict, s):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if 'bn' in k:
            if s in k:
                newkey = k.replace(s, '')
                new_state_dict[newkey]=v
        else:
            new_state_dict[k]=v
    return new_state_dict


def load_advckpt(model, checkpoint_path, s, use_ema):
    print("Load checkpoint from: ", checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if use_ema:
        state_dict=state_dict['state_dict_ema']
    else:
        state_dict=state_dict['state_dict']
    state_dict=convert_model(state_dict, s)
    model.load_state_dict(state_dict, strict=True)
    
    
def print_coco_results(results):
    
    def _print(result, ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(.5, .95) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, result))