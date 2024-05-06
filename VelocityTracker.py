"""
Definition of the VelocityTracker class.
"""

import numpy as np

from Phase import Phase
from RunningAverage import RunningAverage


VELOCITY_COUNT_THRESHOLDS = 3


class VelocityTracker(object):
    def __init__(self, plate_diameter, diff_threshold=0.6, min_distance=0.1):
        """
        Processes measurements resulting in a list
        of Phase objects holding the information about
        each concentric and eccentric phase performed.

        Takes in the plate diameter (in meters) used in the video to transform
        normalized image coordinates into meters and the minimal distance (in meters)
        to filter out some initial noises.
        """

        self.plate_diameter = plate_diameter
        self.min_distance = min_distance
        self.diff_threshold = diff_threshold

        # Initialize variables needed for measurement processing
        self.current_phase = Phase.HOLD
        self.phases = []
        self.max_y_diff = None
        self.y_prev = None
        self.x_prev = None

        self.ys = []
        self.xs = []
        self.widths = []
        self.heights = []
        self.times = []

        self.width_avg = RunningAverage(window_size=30)
        self.width_avg = RunningAverage(window_size=30)

        self.negative_vel_cnt = 0
        self.positive_vel_cnt = 0

    def _filter_phases(self):
        """
        Check a list of Phase objects and based on the
        range of motion in Y axis filter out the exercise
        setup and rerack.
        """

        def del_list_by_indexes(list, is_to_remove):
            return [el for i, el in enumerate(list) if i not in is_to_remove]

        diff_treshold = self.max_y_diff / 2
        is_to_remove = set()

        for i, phase in enumerate(self.phases):
            if phase.y_diff < diff_treshold:
                is_to_remove.add(i)

        self.phases = del_list_by_indexes(self.phases, is_to_remove)

    def _append_to_bar_path(self, x, y, width, height, time):
        """
        Appends current x, y position, normalized plate width and height
        and time to lists for the given phase.
        """

        self.xs.append(x)
        self.ys.append(y)
        self.widths.append(width)
        self.heights.append(height)
        self.times.append(time)

    def _reset_bar_path(self):
        """
        Reset the metrics tracked during an active phase.
        """

        self.xs = []
        self.ys = []
        self.widths = []
        self.heights = []
        self.times = []

    def process_measurements(self, time, x, y, dx, dy, norm_plate_height, norm_plate_width):
        """
        Process a the measured position, velocity and plate dimension
        for a given time point.
        """

        width = self.width_avg.update(norm_plate_width)
        height = self.width_avg.update(norm_plate_height)

        if self.y_prev is not None:
            dy = y - self.y_prev

        if self.current_phase != Phase.HOLD:
            self._append_to_bar_path(
                x, y, width, height, time)

        if self.current_phase == Phase.CONCENTRIC:
            if dy > 0:
                self.positive_vel_cnt += 1
                self.negative_vel_cnt = 0

                if self.positive_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                    self._end_phase()
            else:
                self.positive_vel_cnt = 0

        if self.current_phase == Phase.ECCENTRIC:
            if dy < 0:
                self.negative_vel_cnt += 1
                self.positive_vel_cnt = 0

                if self.negative_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                    self._end_phase()
            else:
                self.negative_vel_cnt = 0
                self.positive_vel_cnt += 1

        if dy < 0 and self.current_phase == Phase.HOLD:
            self.negative_vel_cnt += 1
            self.positive_vel_cnt = 0

            if self.negative_vel_cnt == 1:
                self._reset_bar_path()
            else:
                # Start appending before the rep officially started
                self._append_to_bar_path(
                    x, y, width, height, time)

            if self.negative_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self._start_phase(Phase.CONCENTRIC)

        if dy > 0 and self.current_phase == Phase.HOLD:
            self.positive_vel_cnt += 1
            self.negative_vel_cnt = 0

            if self.positive_vel_cnt == 1:
                self._reset_bar_path()
            else:
                # Start appending before the rep officially started
                self._append_to_bar_path(
                    x, y, width, height, time)

            if self.positive_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self._start_phase(Phase.ECCENTRIC)

        self.x_prev = x
        self.y_prev = y

    def _start_phase(self, phase):
        """
        Set attributes values when either
        an eccentric or a concentric
        phase starts.
        """

        self.current_phase = phase
        self.positive_vel_cnt = 0
        self.negative_vel_cnt = 0

    def _end_phase(self):
        """
        End the processing of a concentric
        or an eccentric phase of the repetition.
        """

        # Find the start/end positions
        if self.current_phase == Phase.CONCENTRIC:
            start_idx = np.argmax(self.ys)
            end_idx = np.argmin(self.ys)
        elif self.current_phase == Phase.ECCENTRIC:
            start_idx = np.argmin(self.ys)
            end_idx = np.argmax(self.ys)

        # Remember the maximal ROM in Y axis
        y_diff = abs(self.ys[start_idx] - self.ys[end_idx])

        if self.max_y_diff is None or y_diff > self.max_y_diff:
            self.max_y_diff = y_diff
            self._filter_phases()

        if y_diff > self.max_y_diff * self.diff_threshold:
            distance = 0

            for i in range(start_idx + 1, end_idx + 1):
                dx = abs(self.xs[i] - self.xs[i-1]) / ((self.widths[i] +
                                                        self.widths[i-1]) / 2) * self.plate_diameter
                dy = abs(self.ys[i] - self.ys[i-1]) / ((self.heights[i] +
                                                        self.heights[i-1]) / 2) * self.plate_diameter

                distance += dx + dy

            if distance < self.min_distance:
                self.negative_vel_cnt = 0
                self.positive_vel_cnt = 0
                self.current_phase = Phase.HOLD
                return

            phase = Phase(
                time_start=self.times[start_idx],
                time_end=self.times[end_idx],
                y_start=self.ys[start_idx],
                y_end=self.ys[end_idx],
                rom=distance,
                phase_type=self.current_phase
            )
            self.phases.append(phase)
            self._filter_phases()

        self.current_phase = Phase.HOLD
        self.positive_vel_cnt = 0
        self.negative_vel_cnt = 0

    def end_processing(self):
        """
        Check if concentric/eccentric phase wasn't still in progress when video ended.
        """

        if self.current_phase != Phase.HOLD:
            self._end_phase()
