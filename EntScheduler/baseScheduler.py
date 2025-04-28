from abc import ABC, abstractmethod


class _EntScheduler(ABC):
    def __init__(self, *, ent_coef):
        self.init_ent_coef = ent_coef
        self.last_epoch = 0

    @abstractmethod
    def get_ent_coef(self):
        """
        Get the entropy value for the current update.
        """
        raise NotImplementedError

    def step(self):
        """
        Step the entropy scheduler.
        """
        self.last_epoch += 1
