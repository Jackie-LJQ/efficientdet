from torch.autograd import Variable
import torch
from utils import set_advState

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
    def __init__(self, eps, alpha=1, targeted=True):
        '''
        Args:
            eps: float. noise bound.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.eps = eps 
        self.targeted = targeted
        self.alpha = alpha
       

    def attack(self, model, x, gtlabels, targets=None):
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
        if self.targeted:
            # total_loss ccls_loss and box_loss of clean sample
            output = model(x_adv, targets)
        else:
            output = model(x_adv, gtlabels)
        cls_loss, box_loss = output["class_loss"], output["box_loss"]
        cls_grad_adv = torch.autograd.grad(cls_loss, x_adv, only_inputs=True, retain_graph=True)[0]
        box_grad_adv = torch.autograd.grad(box_loss, x_adv, only_inputs=True)[0]
        
        with torch.no_grad():
            x_cls_adv = x_adv.data.add_(self.alpha * torch.sign(cls_grad_adv.data)) # gradient assend by Sign-SGD
            x_cls_adv = linf_clamp(x_cls_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_cls_adv = torch.clamp(x_cls_adv, 0, 1) # clamp to RGB range [0,1]
            
            x_box_adv = x_adv.data.add_(self.alpha * torch.sign(box_grad_adv.data)) # gradient assend by Sign-SGD
            x_box_adv = linf_clamp(x_box_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
            x_box_adv = torch.clamp(x_box_adv, 0, 1) # clamp to RGB range [0,1]
            
            # total_loss of cls_adv sample and box_adv sample
            set_advState(model, False)
            cls_loss = model(x_cls_adv, gtlabels)['loss']
            box_loss = model(x_box_adv, gtlabels)['loss']
                
        # set_advState(model, "clean")
        if box_loss > cls_loss:
            return x_box_adv
                
        return x_cls_adv