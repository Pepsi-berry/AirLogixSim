from .baseScheduler import _EntScheduler


class LinearDecayEntCoef(_EntScheduler):
    """
    Linear decay entropy scheduler.
    """

    def __init__(self, *, ent_coef, n_updates):
        super().__init__(ent_coef=ent_coef)
        self.n_updates = n_updates

    def get_ent_coef(self):
        """
        Get the entropy value for the current update.
        """
        return self.init_ent_coef * (1.0 - self.last_epoch / self.n_updates)
