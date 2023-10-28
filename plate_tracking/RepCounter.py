# TODO: Fine tune the treshold value.
VELOCITY_TRESHOLD = 0.03


class Phase(object):
    CONCENTRIC = 0
    ECCENTRIC = 1
    HOLD = 2

    def __init__(self, time_start, time_end, y_start, y_end, phase_type):
        self.time_start = time_start
        self.time_end = time_end
        self.y_start = y_start
        self.y_end = y_end
        self.type = phase_type

    @property
    def y_diff(self):
        return abs(self.y_start - self.y_end)

    def __str__(self):
        if self.type == Phase.CONCENTRIC:
            phase_type = 'concentric'
        elif self.type == Phase.ECCENTRIC:
            phase_type = 'eccentric'
        else:
            phase_type = 'hold'

        return f'{phase_type}, t_start: {self.time_start}, t_end: {self.time_end}, y_start: {self.y_start}, y_end: {self.y_end}'


class RepCounter(object):
    def __init__(self):
        self.current_phase = Phase.HOLD
        self.phases = []
        self.max_y_diff = None

    def _del_list_by_indexes(self, list, is_to_remove):
        return [el for i, el in enumerate(list) if i not in is_to_remove]

    def _filter_concentrics(self):
        """
        Check previous concetrics phases and their range of motion
        to filter out exercise setup.
        """
        diff_treshold = self.max_y_diff / 2
        is_to_remove = set()

        for i, phase in enumerate(self.phases):
            if phase.type == Phase.CONCENTRIC and phase.y_diff < diff_treshold:
                is_to_remove.add(i)

        self.phases = self._del_list_by_indexes(self.phases, is_to_remove)

    def _filter_eccentrics(self):
        """
        Check previous eccentric phases and their range of motion
        to filter out exercise setup.
        """
        diff_treshold = self.max_y_diff / 2
        is_to_remove = set()

        for i, phase in enumerate(self.phases):
            if phase.type == Phase.ECCENTRIC and phase.y_diff < diff_treshold:
                is_to_remove.add(i)

        self.phases = self._del_list_by_indexes(self.phases, is_to_remove)

    def process_measurements(self, time, x, y, dx, dy):
        if dy < -VELOCITY_TRESHOLD and self.current_phase == Phase.HOLD:
            self.time_start = time
            self.y_start = y
            self.current_phase = Phase.CONCENTRIC

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
                    phase_type=self.current_phase
                )

                self.phases.append(phase)
                self._filter_concentrics()

            self.current_phase = Phase.HOLD

        if dy > VELOCITY_TRESHOLD and self.current_phase == Phase.HOLD:
            self.time_start = time
            self.y_start = y
            self.current_phase = Phase.ECCENTRIC

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
                    phase_type=self.current_phase
                )
                self.phases.append(phase)
                self._filter_eccentrics()

            self.current_phase = Phase.HOLD

    def end_processing(self, time, x, y, dx, dy):
        """
        Check if concentric phase wasn't still in progress when video ended.
        """
        if self.current_phase == Phase.CONCENTRIC:
            self.timestamps.append((self.time_start, time))
            self.ys.append((self.y_start, y))
            self.current_phase = Phase.HOLD


def find_concentrics_in_df(df):
    rep_counter = RepCounter()

    for _, (time, _, _, x, y, dx, dy, _) in df.iterrows():
        rep_counter.process_measurements(time, x, y, dx, dy)

    rep_counter.end_processing(time, x, y, dx, dy)

    return rep_counter.phases
