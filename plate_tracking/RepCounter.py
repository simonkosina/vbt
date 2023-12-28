import numpy as np

from Phase import Phase


# TODO: Increase the toehold to 0.05 and go back through the data to find the last highest y pos
#       position before the rep started. Should work to filter the end of concentric phase.
#       To improve the accuracy of finding the concentric starting point (we could remember the
#       last time the velocity crossed zero and use it if the velocity crosses some threshold.
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
        self.acc_dist_x = 0
        self.acc_dist_y = 0
        self.y_prev = None
        self.x_prev = None
        
        # TODO: Finish the rep counting offset
        self.y_max_value = -np.inf
        self.y_min_value = np.inf
        self.y_max_time = None
        self.y_min_time = None

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
        if dy < 0 and self.current_phase == Phase.HOLD:
            self.negative_vel_cnt += 1
            self.positive_vel_cnt = 0

            if self.negative_vel_cnt == 1:
                self.time_start = time
                self.y_start = y

            if self.negative_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self.start_phase(Phase.CONCENTRIC)

        if dy > 0 and self.current_phase == Phase.HOLD:
            self.positive_vel_cnt += 1
            self.negative_vel_cnt = 0

            if self.positive_vel_cnt == 1:
                self.time_start = time
                self.y_start = y

            if self.positive_vel_cnt >= VELOCITY_COUNT_THRESHOLDS:
                self.start_phase(Phase.ECCENTRIC)

        if dy > 0 and self.current_phase == Phase.CONCENTRIC:
            self.end_phase(y, time, norm_plate_width, norm_plate_height)

        if dy < 0 and self.current_phase == Phase.ECCENTRIC:
            self.end_phase(y, time, norm_plate_width, norm_plate_height)

        if self.x_prev is not None and self.y_prev is not None:
            self.acc_dist_x += abs(x - self.x_prev)
            self.acc_dist_y += abs(y - self.y_prev)

        self.x_prev = x
        self.y_prev = y

    def start_phase(self, phase):
        self.current_phase = phase
        self.acc_dist_x = 0
        self.acc_dist_y = 0
        self.positive_vel_cnt = 0
        self.negative_vel_cnt = 0

    def end_phase(self, y, time, norm_plate_width, norm_plate_height):
        # Remember the maximal ROM in Y axis
        y_diff = abs(self.y_start - y)
        if self.max_y_diff is None or y_diff > self.max_y_diff:
            self.max_y_diff = y_diff

        if y_diff > self.max_y_diff / 2:
            rom = (self.acc_dist_x / norm_plate_width) * self.plate_diameter + \
                (self.acc_dist_y / norm_plate_height) * self.plate_diameter
            phase = Phase(
                time_start=self.time_start,
                time_end=time,
                y_start=self.y_start,
                y_end=y,
                rom=rom,
                phase_type=self.current_phase
            )
            self.phases.append(phase)
            self._filter_phases()

        self.current_phase = Phase.HOLD

    def end_processing(self, time, x, y, dx, dy, norm_plate_height, norm_plate_width):
        """
        Check if concentric/eccentric phase wasn't still in progress when video ended.
        """
        # TODO:
        if self.current_phase == Phase.CONCENTRIC:
            pass
        elif self.current_phase == Phase.ECCENTRIC:
            pass


def find_concentrics_in_df(df, plate_diameter):
    rep_counter = RepCounter(plate_diameter)

    for _, (time, x, y, dx, dy, norm_plate_height, norm_plate_width) in df.iterrows():
        rep_counter.process_measurements(
            time, x, y, dx, dy, norm_plate_height, norm_plate_width)

    rep_counter.end_processing(
        time, x, y, dx, dy, norm_plate_height, norm_plate_width)

    return rep_counter.phases
