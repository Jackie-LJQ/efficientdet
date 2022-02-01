import torch.nn as nn
import torch
class Det_Dual_BN(nn.Module):
    def __init__(self, num_feature, eps=1e-3, momentum=0.1, affine=True, track_running_stats=True):
        super(Det_Dual_BN, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum, affine,track_running_stats)
        self.bn_adv = nn.BatchNorm2d(num_feature, eps, momentum, affine, track_running_stats)
        self.advState = True #Trur for clean, False for adversarial sample
    def forward(self, input):
        if self.advState:
            output = self.bn_clean(input)
        else:
            output = self.bn_adv(input)
        return output
    def __repr__(self):
        return "auxiliary_bn dual bn"
        