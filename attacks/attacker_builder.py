from .fgsm import *

__all__ = ["AttackerBuilder"]
def AttackerBuilder(attacker_name='FGSM'):
    if attacker_name == "FGSM":
        args = dict(eps=8/255, targeted=False)
        # if attacker_cfg.get("cfg"):
        #     args.update(attacker_cfg["cfg"])
        attacker = FGSM(**args)
    else:
        assert False, "Unknown attacker type %s" % attacker_name
    return attacker
        
    
    
    