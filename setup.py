import os
import sys

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Ensure these are present (in case we are not using PEP-518 compatible build
# system).
import setuptools_scm

use_avx2 = os.environ.get("SCALIB_AVX2") is not None

with open("src/scalib/build_config.py", "w") as f:
    f.write(f"REQUIRE_AVX2 = {use_avx2}\n")

rustc_flags = []
if use_avx2:
    rustc_flags.extend(["-C", "target-feature=+avx2"])

scalib_features = ["pyo3/abi3"]

if sys.platform == "linux":
    scalib_features.append("blis")

setup(
    project_urls={
        "Bug Tracker": "https://github.com/simple-crypto/scalib/issues",
    },
    rust_extensions=[
        RustExtension(
            "scalib._scalib_ext",
            path="src/scalib_ext/scalib-py/Cargo.toml",
            binding=Binding.PyO3,
            features=scalib_features,
            py_limited_api=True,
            rustc_flags=rustc_flags,
        )
    ],
)
