from .baseScheduler import _EntScheduler
import math


class CosineAnnealingEntCoef(_EntScheduler):
    def __init__(self, ent_coef, T_max, eta_min=0.0):
        """
        Cosine annealing entropy scheduler.

        Args:
            ent_coef (float): Initial entropy coefficient (eta_0).
            T_max (int): Number of steps for a full cosine cycle.
            eta_min (float): Minimum entropy coefficient.
        """
        super().__init__(ent_coef=ent_coef)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_ent_coef(self):
        """
        Get the entropy coefficient using cosine annealing schedule.
        """
        T_cur = self.last_epoch
        cosine_decay = 0.5 * (1 + math.cos(math.pi * T_cur / self.T_max))
        ent_coef = self.eta_min + (self.init_ent_coef - self.eta_min) * cosine_decay
        return ent_coef