import torch.nn as nn
import torch
class Triple_BN(nn.Module):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1):
        super(Triple_BN, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_cls = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_box = nn.BatchNorm2d(num_feature, eps, momentum)
    def forward(self, input):
        batch_size = input.shape[0] // 3
        clean, adv_cls, adv_box = torch.split(input, [batch_size, batch_size, batch_size], dim=0)
        clean_out = self.bn_clean(clean)
        cls_adv_out = self.bn_cls(adv_cls)
        box_adv_out = self.bn_box(adv_box)
        output = torch.cat([clean_out, cls_adv_out, box_adv_out], dim=0)
        return output
    def __repr__(self):
        return "auxiliary_bn triple bn"
        