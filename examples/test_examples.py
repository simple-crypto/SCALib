"""
Module only used by pytest.
"""

import pathlib
import runpy


def example_path(example_file):
    return str(pathlib.Path(__file__).parent / example_file)


def test_examples():
    for p in ["aes_attack.py", "aes_tvla.py", "aes_info.py"]:
        runpy.run_path(example_path(p))
