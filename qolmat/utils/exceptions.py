class KerasExtraNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install keras xx.xx.xx
        pip install qolmat[keras]"""
        )


class SignalTooShort(Exception):
    def __init__(self, period, n_cols):
        super().__init__(
            f"""`period` must be smaller than the signals duration.
            `period`is {period} but the number of columns if {n_cols}"""
        )
