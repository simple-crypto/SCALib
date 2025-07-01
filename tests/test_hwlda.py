import pickle

import pytest
import numpy as np
import scipy.linalg
import scipy.special

from scalib import ScalibError
from scalib.modeling import HwLda, HwLdaAcc

from utils_test import get_rng

import copy


def test_run_hwlda():
    hwlda = HwLdaAcc(4)
    assert True
