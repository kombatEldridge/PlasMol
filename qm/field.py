# field.py
import numpy as np

class FIELD():
    def __init__(self):
        self.exc_store = {}
        self.empty = np.array([0.0, 0.0, 0.0])

    def get_exc_t_plus_dt(self):
        key = 'exc_t_plus_dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_plus_dt(self, exc_t_plus_dt):
        self.exc_store['exc_t_plus_dt'] = exc_t_plus_dt

    def get_exc_t(self):
        key = 'exc_t'
        return self.exc_store.get(key, self.empty)

    def set_exc_t(self, exc_t):
        self.exc_store['exc_t'] = exc_t

    def get_exc_t_minus_dt(self):
        key = 'exc_t_minus_dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_minus_dt(self, exc_t_minus_dt):
        self.exc_store['exc_t_minus_dt'] = exc_t_minus_dt

    def get_exc_t_minus_2dt(self):
        key = 'exc_t_minus_2dt'
        return self.exc_store.get(key, self.empty)

    def set_exc_t_minus_2dt(self, exc_t_minus_2dt):
        self.exc_store['exc_t_minus_2dt'] = exc_t_minus_2dt
