from Phase import Phase


# TODO: Fine tune the treshold value.
VELOCITY_TRESHOLD = 0.03


class RepCounter(object):
    def __init__(self, plate_diameter):
        """
        Processes meassurements resulting in a list
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
        self.acc_distance = 0
        self.y_prev = None
        self.x_prev = None

    def _filter_phases(self, phase_type):
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
            if phase.type == phase_type and phase.y_diff < diff_treshold:
                is_to_remove.add(i)

        self.phases = del_list_by_indexes(self.phases, is_to_remove)

    def _filter_concentrics(self):
        """
        Check previous concetric phases and based on the
        range of motion in Y axis to filter out exercise
        and rerack.
        """
        self._filter_phases(Phase.CONCENTRIC)

    def _filter_eccentrics(self):
        """
        Check previous eccentric phases and based on the
        range of motion in Y axis to filter out exercise
        and rerack.
        """
        self._filter_phases(Phase.ECCENTRIC)

    def process_measurements(self, time, x, y, dx, dy, norm_diameter):
        if dy < -VELOCITY_TRESHOLD and self.current_phase == Phase.HOLD:
            self.time_start = time
            self.y_start = y
            self.current_phase = Phase.CONCENTRIC
            self.acc_distance = 0

        if dy > 0 and self.current_phase == Phase.CONCENTRIC:
            # Remember the maximal ROM in Y axis
            y_diff = abs(self.y_start - y)
            if self.max_y_diff is None or y_diff > self.max_y_diff:
                self.max_y_diff = y_diff

            if y_diff > self.max_y_diff / 2:
                phase = Phase(
                    time_start=self.time_start,
                    time_end=time,
                    y_start=self.y_start,
                    y_end=y,
                    rom=self.acc_distance / norm_diameter * self.plate_diameter,
                    phase_type=self.current_phase
                )

                self.phases.append(phase)
                self._filter_concentrics()

            self.current_phase = Phase.HOLD

        if dy > VELOCITY_TRESHOLD and self.current_phase == Phase.HOLD:
            self.time_start = time
            self.y_start = y
            self.current_phase = Phase.ECCENTRIC
            self.acc_distance = 0

        if dy < 0 and self.current_phase == Phase.ECCENTRIC:
            # Remember the maximal ROM in Y axis
            y_diff = abs(self.y_start - y)
            if self.max_y_diff is None or y_diff > self.max_y_diff:
                self.max_y_diff = y_diff

            if y_diff > self.max_y_diff / 2:
                phase = Phase(
                    time_start=self.time_start,
                    time_end=time,
                    y_start=self.y_start,
                    y_end=y,
                    rom=self.acc_distance / norm_diameter * self.plate_diameter,
                    phase_type=self.current_phase
                )
                self.phases.append(phase)
                self._filter_eccentrics()

            self.current_phase = Phase.HOLD

        if self.x_prev is not None and self.y_prev is not None:
            self.acc_distance += ((x - self.x_prev)**2 +
                                  (y - self.y_prev)**2)**0.5

        self.x_prev = x
        self.y_prev = y

    def end_processing(self, time, x, y, dx, dy, norm_diameter):
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

    for _, (time, _, _, x, y, dx, dy, norm_diameter) in df.iterrows():
        rep_counter.process_measurements(time, x, y, dx, dy, norm_diameter)

    rep_counter.end_processing(time, x, y, dx, dy, norm_diameter)

    return rep_counter.phases
