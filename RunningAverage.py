"""
Implements the RunningAverage class which is used to calculate the running
average of a sequence of numbers.
"""

from collections import deque


class RunningAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.window = deque()
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.window.append(value)
        self.total += value
        self.count += 1

        if self.count >= self.window_size:
            average = self.total / self.window_size
            self.total -= self.window.popleft()
            self.count -= 1
            return average
        else:
            return self.total / self.count
