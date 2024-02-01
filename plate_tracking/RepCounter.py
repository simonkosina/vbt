import numpy as np

from Phase import Phase


VELOCITY_COUNT_THRESHOLDS = 3


class RepCounter(object):
    def __init__(self, plate_diameter):
        """
        Processes measurements resulting in a list
        of Phase objects holding the information about
        each concentric and eccentric phase performed.

        Takes in the plate diameter (in meters) used in the video to transform
        normalized image coordinates into meters.
        """
        self.plate_diameter = plate_diameter

        # Initialize variables needed for measurement processing
        self.current_phase = Phase.HOLD
        self.phases = []
        self.max_y_diff = None
        self.y_prev = None
        self.x_prev = None

        self.ys = []
        self.xs = []
        self.times = []

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

    def process_measurements(self, time, x, y, dx, dy, norm_plate_height, norm_plate_width):
        if self.y_prev is not None:
            dy = y - self.y_prev

        if self.current_phase == Phase.CONCENTRIC:
            if dy > 0:
                self.positive_vel_cnt += 1
                self.negative_vel_cnt = 0

                if self.positive_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                    self.end_phase(norm_plate_width, norm_plate_height)
            else:
                self.positive_vel_cnt = 0

        if self.current_phase == Phase.ECCENTRIC:
            if dy < 0:
                self.negative_vel_cnt += 1
                self.positive_vel_cnt = 0

                if self.negative_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                    self.end_phase(norm_plate_width, norm_plate_height)
            else:
                self.negative_vel_cnt = 0
                self.positive_vel_cnt += 1

        if dy < 0 and self.current_phase == Phase.HOLD:
            self.negative_vel_cnt += 1
            self.positive_vel_cnt = 0

            if self.negative_vel_cnt == 1:
                self.ys = [y]
                self.xs = [x]
                self.times = [time]
            else:
                # Start appending before the rep officially started
                self.ys.append(y)
                self.xs.append(x)
                self.times.append(time)

            if self.negative_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self.start_phase(Phase.CONCENTRIC)

        if dy > 0 and self.current_phase == Phase.HOLD:
            self.positive_vel_cnt += 1
            self.negative_vel_cnt = 0

            if self.positive_vel_cnt == 1:
                self.ys = [y]
                self.xs = [x]
                self.times = [time]
            else:
                # Start appending before the rep officially started
                self.ys.append(y)
                self.xs.append(x)
                self.times.append(time)

            if self.positive_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self.start_phase(Phase.ECCENTRIC)

        if self.current_phase != Phase.HOLD:
            self.ys.append(y)
            self.xs.append(x)
            self.times.append(time)

        self.x_prev = x
        self.y_prev = y

    def start_phase(self, phase):
        self.current_phase = phase
        self.positive_vel_cnt = 0
        self.negative_vel_cnt = 0

    def end_phase(self, norm_plate_width, norm_plate_height):
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

        if y_diff > self.max_y_diff / 2:
            acc_dist_x = 0
            acc_dist_y = 0

            for i in range(start_idx + 1, end_idx + 1):
                acc_dist_x += abs(self.xs[i] - self.xs[i-1])
                acc_dist_y += abs(self.ys[i] - self.ys[i-1])

            rom = (acc_dist_x / norm_plate_width) * self.plate_diameter + (acc_dist_y / norm_plate_height) * self.plate_diameter
            phase = Phase(
                time_start=self.times[start_idx],
                time_end=self.times[end_idx],
                y_start=self.ys[start_idx],
                y_end=self.ys[end_idx],
                rom=rom,
                phase_type=self.current_phase
            )
            self.phases.append(phase)
            self._filter_phases()

        self.current_phase = Phase.HOLD
        self.positive_vel_cnt = 0
        self.negative_vel_cnt = 0

    def end_processing(self, time, x, y, dx, dy, norm_plate_height, norm_plate_width):
        """
        Check if concentric/eccentric phase wasn't still in progress when video ended.
        """
        if self.current_phase != Phase.HOLD:
            self.end_phase(norm_plate_width, norm_plate_height)


def find_concentrics_in_df(df, plate_diameter):
    rep_counter = RepCounter(plate_diameter)

    for _, (time, x, y, dx, dy, norm_plate_height, norm_plate_width) in df.iterrows():
        rep_counter.process_measurements(
            time, x, y, dx, dy, norm_plate_height, norm_plate_width)

    rep_counter.end_processing(
        time, x, y, dx, dy, norm_plate_height, norm_plate_width)

    return rep_counter.phases
