import torch.optim as optim
from lion_pytorch import Lion
from adamp import AdamP

class OptimizerSelector:
    def __init__(self, opt, model, **kwargs):
        self.opt = opt
        self.model = model
        self.params = kwargs
    
    def get_optim(self):
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(params=self.model.parameters(), **self.params)
        elif self.opt == 'AdamP':
            self.optimizer = AdamP(params=self.model.parameters(), **self.params)
        elif self.opt == 'Lion':
            self.optimizer = Lion(params=self.model.parameters(), **self.params)
        return self.optimizer


# optim = OptimizerSelector(cfg.optim, model, cfg.lr, cfg.weight_decay)
# 예시) optimizer_selector = OptimizerSelector('adam', model, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
# optimizer = optim.get_optim()