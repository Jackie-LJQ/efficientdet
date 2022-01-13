import torch.nn as nn
import torch
class Triple_BN(nn.Module):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1):
        super(Triple_BN, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_cls = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_box = nn.BatchNorm2d(num_feature, eps, momentum)
        self.advState = "clean" #clean, cls_adv, box_adv
    def forward(self, input):
        if self.advState == "clean":
            output = self.bn_clean(input)
        elif self.advState == "cls_adv":
            output = self.bn_cls(input)
        else:
            output = self.bn_box(input)
        return output
    def __repr__(self):
        return "auxiliary_bn triple bn"
        