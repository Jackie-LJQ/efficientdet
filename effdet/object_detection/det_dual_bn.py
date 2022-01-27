import torch.nn as nn
import torch
class Det_Dual_BN(nn.Module):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(Det_Dual_BN, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum, affine,track_running_stats)
        self.bn_cls = nn.BatchNorm2d(num_feature, eps, momentum, affine, track_running_stats)
        self.bn_box = nn.BatchNorm2d(num_feature, eps, momentum, affine, track_running_stats)
        self.advState = 1 #1 for clean, 2 for class adv sample , 3 for box adversarial sample
    def forward(self, input):
        if self.advState==1:
            output = self.bn_clean(input)
        elif self.advState==2:
            output = self.bn_cls(input)
        else:
            output = self.bn_box(input)
        return output
    def __repr__(self):
        return "auxiliary_bn dual bn"
        