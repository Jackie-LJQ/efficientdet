from torch.autograd import Variable
import torch


def linf_clamp(x, _min, _max):
    '''
    Inplace linf clamping on Tensor x.

    Args:
        x: Tensor. shape=(N,C,W,H)
        _min: Tensor with same shape as x.
        _max: Tensor with same shape as x.
    '''
    idx = x.data < _min
    x.data[idx] = _min[idx]
    idx = x.data > _max
    x.data[idx] = _max[idx]

    return x

class FGSM():
    def __init__(self, eps, targeted=False):
        '''
        Args:
            eps: float. noise bound.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.eps = eps 
        self.targeted = targeted
       

    def attack(self, model, x, gtlabels=None, targets=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            gtlabels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        x_adv = Variable(x.detach().cuda(), requires_grad=True)
        dummy_x = torch.cat([x_adv, torch.zeros_like(x_adv), torch.zeros_like(x_adv)], dim=0)
        if self.targeted:
            # total_loss ccls_loss and box_loss of clean sample
            total_loss, cls_loss, box_loss = model(dummy_x, targets)[0].values()
        else:
            total_loss, cls_loss, box_loss = model(dummy_x, gtlabels)[0].values()
        cls_grad_adv = torch.autograd.grad(cls_loss, x_adv, only_inputs=True, retain_graph=True)[0]
        box_grad_adv = torch.autograd.grad(box_loss, x_adv, only_inputs=True)[0]
        
        x_cls_adv = x_adv.data.add_(self.eps * torch.sign(cls_grad_adv.data)) # gradient assend by Sign-SGD
        x_cls_adv = linf_clamp(x_cls_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
        x_cls_adv = torch.clamp(x_cls_adv, 0, 1) # clamp to RGB range [0,1]
        
        x_box_adv = x_adv.data.add_(self.eps * torch.sign(box_grad_adv.data)) # gradient assend by Sign-SGD
        x_box_adv = linf_clamp(x_box_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
        x_box_adv = torch.clamp(x_box_adv, 0, 1) # clamp to RGB range [0,1]
        
        cat_input = torch.cat([x, x_cls_adv, x_box_adv], dim=0)
        # total_loss of cls_adv sample and box_adv sample
        cat_loss = model(cat_input, gtlabels)
        cls_loss = cat_loss[1]['loss']
        box_loss = cat_loss[2]['loss']        
        
        if box_loss > cls_loss:
            return x_box_adv, 'box'
                
        return x_cls_adv, 'cls'