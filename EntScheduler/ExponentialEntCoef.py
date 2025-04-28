from .baseScheduler import _EntScheduler


class ExponentialEntCoef(_EntScheduler):
    """
    Exponential entropy scheduler.
    """

    def __init__(self, *, ent_coef, decay=0.99):
        super().__init__(ent_coef=ent_coef)
        self.decay_rate = decay

    def get_ent_coef(self):
        """
        Get the entropy value for the current update.
        """
        return self.init_ent_coef * (self.decay_rate ** self.last_epoch)
