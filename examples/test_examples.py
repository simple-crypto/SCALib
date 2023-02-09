"""
Module only used by pytest.
"""

import pathlib
import runpy


def example_path(example_file):
    return str(pathlib.Path(__file__).parent / example_file)


def test_examples():
    runpy.run_path(example_path("aes_simulation.py"))
