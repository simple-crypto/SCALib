
# Based on https://scikit-hep.org/developer/packaging

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Ensure these are present (in case we are not using PEP-518 compatible build
# system).
import setuptools_scm
import toml


setup(
    project_urls={
        "Bug Tracker": "https://github.com/obronchain/scaffe/issues",
    },
    rust_extensions=[
        RustExtension(
            "scaffe._scaffe_ext",
            path="src/scaffe_ext/scaffe-py/Cargo.toml",
            binding=Binding.PyO3,
            features=["pyo3/abi3"],
            py_limited_api=True,
        )
    ],
)
