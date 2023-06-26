class PyTorchExtraNotInstalled(Exception):
    def __init__(self):
        super().__init__(
            """Please install torch xx.xx.xx
        pip install qolmat[torch]"""
        )
