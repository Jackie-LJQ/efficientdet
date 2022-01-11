
class FGSM():
    def __init__(self, eps, targeted=False):
        '''
        Args:
            eps: float. noise bound.
            targeted: bool. If Ture, do targeted attack.
        '''
        self.eps = eps 
        self.targeted = targeted
       

    def attack(self, model, x, gtlabels=None, targets=None, _lambda=None, idx2BN=None):
        '''
        Args:
            x: Tensor. Original images. size=(N,C,W,H)
            model: nn.Module. The model to be attacked.
            gtlabels: Tensor. ground truth labels for x. size=(N,). Useful only under untargeted attack.
            targets: Tensor. target attack class for x. size=(N,). Useful only under targeted attack.

        Return:
            x_adv: Tensor. Adversarial images. size=(N,C,W,H)
        '''
        
        if self.targeted:
            total_loss, cls_loss, box_loss = model(x, targets).values()
        else:
            total_loss, cls_loss, box_loss = model(x, gtlabels).values()
        cls_grad_adv = torch.autograd.grad(cls_loss, x_adv, only_inputs=True)[0]
        box_grad_adv = torch.autograd.grad(box_loss, x_adv, only_inputs=True)[0]
        
        x_cls_adv.data.add_(self.alpha * torch.sign(cls_grad_adv.data)) # gradient assend by Sign-SGD
        x_cls_adv = linf_clamp(x_cls_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
        x_cls_adv = torch.clamp(x_cls_adv, 0, 1) # clamp to RGB range [0,1]
        x_cls_adv.sample_type='adv_cls'
        cls_loss = model(x_cls_adv, gtlabels)['loss']
        
        x_box_adv.data.add_(self.alpha * torch.sign(box_grad_adv.data)) # gradient assend by Sign-SGD
        x_box_adv = linf_clamp(x_box_adv, _min=x-self.eps, _max=x+self.eps) # clamp to linf ball centered by x
        x_box_adv = torch.clamp(x_box_adv, 0, 1) # clamp to RGB range [0,1]
        x_box_adv.sample_type='adv_box'
        box_loss = model(x_box_adv, gtlabels)['loss']
        
        if box_loss > cls_loss:
            return x_box_adv
                
        return x_cls_adv