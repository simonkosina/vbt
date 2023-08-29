import numpy as np

from enum import Enum
from math import floor

HEIGHT_TRESHOLD = 0.025


class Phase(Enum):
    CONCENTRIC = 0
    ECCENTRIC = 1
    HOLD = 2


class RepCounter(object):
    @staticmethod
    def calculate_min_treshold(height_min):
        return height_min + HEIGHT_TRESHOLD

    @staticmethod
    def calculate_max_treshold(height_max):
        return height_max - HEIGHT_TRESHOLD

    @staticmethod
    def opposite_phase(phase):
        if phase == Phase.CONCENTRIC:
            return Phase.ECCENTRIC
        elif phase == Phase.ECCENTRIC:
            return Phase.CONCENTRIC

    def __init__(self, starting_phase):
        if starting_phase not in [Phase.CONCENTRIC, Phase.ECCENTRIC]:
            raise ValueError('Starting phase must be either concentri or eccentric')

        self.starting_phase = starting_phase
        self.curr_phase = Phase.HOLD
        self.prev_phase = self.opposite_phase(starting_phase)

        self.first_rep = True
        self.height = None
        self.min_treshold = np.inf
        self.max_treshold = -np.inf

        self.num_holds = 0

        self.height_min = np.inf
        self.height_max = -np.inf

        self.concentric_start = False
        self.concentric_end = False


    @property
    def num_reps(self):
        return floor(self.num_holds/2)

    # FIXME: Need to end to first rep sooner, not after it already started dropping!
    def _first_concentric_rep(self, height):
        if self.curr_phase == Phase.HOLD and height < self.max_treshold:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.prev_phase = Phase.HOLD
            self.concentric_start = self.curr_phase == Phase.CONCENTRIC

        if self.curr_phase != Phase.HOLD and height > self.min_treshold:
            self.concentric_end = self.curr_phase == Phase.CONCENTRIC
            self.curr_phase = self.opposite_phase(self.curr_phase)
            self.prev_phase = Phase.HOLD
            self.first_rep = False
            self.num_holds += 1

    def _first_eccentric_rep(self, height):
        if self.curr_phase == Phase.HOLD and height > self.min_treshold:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.prev_phase = Phase.HOLD
            self.concentric_end = self.curr_phase == Phase.CONCENTRIC

        if self.curr_phase != Phase.HOLD and height < self.max_treshold:
            self.concentric_end = self.curr_phase == Phase.CONCENTRIC
            self.curr_phase = self.opposite_phase(self.curr_phase)
            self.prev_phase = Phase.HOLD
            self.concentric_end = True
            self.first_rep = False
            self.num_holds += 1

    def update_min_treshold(self, height):
        if self.height_min > height:
            self.height_min = height

        new_treshold = self.calculate_min_treshold(height)

        if self.min_treshold > new_treshold:
            self.min_treshold = new_treshold

    def update_max_treshold(self, height):
        if self.height_max < height:
            self.height_max = height

        new_treshold = self.calculate_max_treshold(height)

        if self.max_treshold < new_treshold:
            self.max_treshold = new_treshold

    def update(self, height):
        self.concentric_start = False
        self.concentric_end = False

        self.update_min_treshold(height)
        self.update_max_treshold(height)

        # Handles first rep
        if self.first_rep and self.starting_phase == Phase.CONCENTRIC:
            self._first_concentric_rep(height)
            return

        if self.first_rep and self.starting_phase == Phase.ECCENTRIC:
            self._first_eccentric_rep(height)
            return

        # Handles subsequent reps
        if height > self.min_treshold and height < self.max_treshold and self.curr_phase == Phase.HOLD:
            self.curr_phase = self.opposite_phase(self.prev_phase)
            self.concentric_start = self.curr_phase == Phase.CONCENTRIC
            self.prev_phase = Phase.HOLD

        if (height < self.min_treshold or height > self.max_treshold) and self.curr_phase != Phase.HOLD:
            self.concentric_end = self.curr_phase == Phase.CONCENTRIC
            self.prev_phase = self.curr_phase
            self.curr_phase = Phase.HOLD
            self.num_holds += 1
