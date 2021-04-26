import os
import numpy as np
import random

import pytest


@pytest.fixture(autouse=True)
def seed_random():
    init1 = random.SystemRandom().randrange(0, 2 ** 32)
    init2 = random.SystemRandom().randrange(0, 2 ** 32)
    print(f"Ranom seeds: random.seed({init1}); np.random.seed({init2})")
    random.seed(init1)
    np.random.seed(init2)
