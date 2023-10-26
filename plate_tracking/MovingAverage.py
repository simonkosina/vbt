import numpy as np


class MovingAverage(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.index = 0
        self.values = []
    
    def process(self, x):
        """
        Process the next value and return
        the current moving average
        """

        if len(self.values) < self.window_size:
            self.values.append(x)
        else:
            self.values[self.index] = x

        self.index = (self.index + 1) % self.window_size

        return np.mean(self.values)
