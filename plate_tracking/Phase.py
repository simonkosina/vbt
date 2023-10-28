class Phase(object):
    CONCENTRIC = 0
    ECCENTRIC = 1
    HOLD = 2

    def __init__(self, time_start, time_end, y_start, y_end, rom, phase_type):
        self.time_start = time_start
        self.time_end = time_end
        self.y_start = y_start
        self.y_end = y_end
        self.type = phase_type
        self.rom = rom # Range of Motion [m]

    @property
    def y_diff(self):
        return abs(self.y_start - self.y_end)

    @property
    def duration(self):
        return self.time_end - self.time_start

    def __str__(self):
        if self.type == Phase.CONCENTRIC:
            phase_type = 'concentric'
        elif self.type == Phase.ECCENTRIC:
            phase_type = 'eccentric'
        else:
            phase_type = 'hold'

        return f'{phase_type}, t_start: {self.time_start}, t_end: {self.time_end}, y_start: {self.y_start}, y_end: {self.y_end}'
