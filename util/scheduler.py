from typing import List
from torch.optim.lr_scheduler import _LRScheduler


class InverseLinearTimeDecay(_LRScheduler):
    """
    Implements an inverse linear time decay learning rate schedule.
    """

    def __init__(self, optimizer, lr, n_updates, last_epoch=-1):
        self.lr = lr
        self.n_updates = n_updates
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute the learning rate based on the current epoch.
        """
        if self.last_epoch >= self.n_updates:
            return [0.0 for _ in self.optimizer.param_groups]

        return [
            self.lr * (1.0 - self.last_epoch / self.n_updates)
            for _ in self.optimizer.param_groups
        ]
