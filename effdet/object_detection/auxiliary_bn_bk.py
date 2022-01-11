import torch.nn as nn
class auxiliary_bn(nn.Module):
    def __init__(self, num_feature, eps=1e-5, momentum=0.1):
        super(auxiliary_bn, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_cls = nn.BatchNorm2d(num_feature, eps, momentum)
        self.bn_box = nn.BatchNorm2d(num_feature, eps, momentum)
    def forward(self, input):
        if input.sample_type=='clean':
            output = self.bn_clean(input)
        elif input.sample_type=='adv_cls':
            output = self.bn_cls(input)
        elif input.sample_type=='adv_box':
            output = self.bn_box(input)
        else:
            assert False, 'Undefinced sample_type %s' % input.sample_type
        output.sample_type = input.sample_type
        return output
    def __repr__(self):
        return "auxiliary_bn"
        