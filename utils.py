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
                model_output.bn_cls.weight = model.weight
                model_output.bn_cls.bias = model.bias
                model_output.bn_box.weight = model.weight
                model_output.bn_box.bias = model.bias
                
        model_output.bn_clean.running_mean = model.running_mean
        model_output.bn_clean.running_var = model.running_var
        model_output.bn_clean.num_batches_tracked = model.num_batches_tracked
        model_output.bn_cls.running_mean = model.running_mean
        model_output.bn_cls.running_var = model.running_var
        model_output.bn_cls.num_batches_tracked = model.num_batches_tracked
        model_output.bn_box.running_mean = model.running_mean
        model_output.bn_box.running_var = model.running_var
        model_output.bn_box.num_batches_tracked = model.num_batches_tracked
    
    for name, child in model.named_children():
        model_output.add_module(name, convert_dual_bn(child))
    
    del model
    return model_output
