from torch.optim import lr_scheduler
from transformers import get_linear_schedule_with_warmup

class SchedulerSelector:
    def __init__(self, sched, optimizer, epoch, batch):
        self.optimizer = optimizer
        self.epoch = epoch
        if sched == 'step': # step size마다 gamma 비율로 lr을 감소
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step=5, gamma=0.6)
        elif sched == 'cosine': # learing rate가 cos함수를 따라서 감소 및 증가
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch, eta_min=1e-6)
        elif sched == 'exp': # learing rate decay가 exponential함수를 따라 감소
            self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
        elif sched == 'plateau': # 성능이 향상이 없을 때 learning rate를 감소
            self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif sched == 'glsww':
            self.scheduler = get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps = 0,
                                                             num_training_steps = batch*epoch)
            
    def get_sched(self):
        return self.scheduler