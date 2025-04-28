from .baseScheduler import _EntScheduler


class StepEntCoef(_EntScheduler):
    def __init__(self, ent_coef, step_size, decay=0.1):
        """
        Step decay entropy scheduler.
        """
        super().__init__(ent_coef=ent_coef)
        self.step_size = step_size
        self.decay = decay

    def get_ent_coef(self):
        """
        Get the entropy value for the current update.
        """
        return self.init_ent_coef * (self.decay ** (self.last_epoch // self.step_size))