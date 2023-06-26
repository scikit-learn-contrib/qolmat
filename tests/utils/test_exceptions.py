import pytest
from qolmat.utils import exceptions


def test_utils_exception_init():
    try:
        raise exceptions.PyTorchExtraNotInstalled()
    except exceptions.PyTorchExtraNotInstalled as e:
        assert (
            str(e)
            == """Please install keras xx.xx.xx
        pip install qolmat[keras]"""
        )
