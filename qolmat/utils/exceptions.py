class KerasExtraNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install keras xx.xx.xx
        pip install qolmat[keras]"""
        )

class PytorchNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install pytorch xx.xx.xx
        pip install qolmat[pytorch]"""
        )