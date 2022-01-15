import torch.nn as nn
import torch
class Det_Dual_BN(nn.Module):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1):
        super(Det_Dual_BN, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_adv = nn.BatchNorm2d(num_feature, eps, momentum)
    def forward(self, input):
        batch_size = input.shape[0] // 3
        clean, cls_adv, box_adv = torch.split(input, [batch_size, batch_size, batch_size], dim=0)
        clean_out = self.bn_clean(clean)
        cls_out = self.bn_adv(cls_adv)
        box_out = self.bn_adv(box_adv)
        output = torch.cat([clean_out, cls_out, box_out], dim=0)
        return output
    def __repr__(self):
        return "auxiliary_bn dual bn"
        