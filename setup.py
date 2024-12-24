import os
import sys

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Ensure these are present (in case we are not using PEP-518 compatible build
# system).
import setuptools_scm


def env_true(env):
    return env == "1"


noflags = env_true(os.environ.get("SCALIB_NOFLAGS"))
portable = not noflags and env_true(os.environ.get("SCALIB_PORTABLE"))
x86_64_v3 = not noflags and env_true(os.environ.get("SCALIB_X86_64_V3"))

if portable and x86_64_v3:
    raise ValueError("Cannot have both SCALIB_PORTABLE and SCALIB_X86_64_V3.")

# We check only for X86_64_V3, as this is the CI default, otherwise we assume local
# builds.
with open("src/scalib/build_config.py", "w") as f:
    f.write(f"REQUIRE_X86_64_V3 = {x86_64_v3}\n")


if noflags or portable:
    rustflags = None
elif x86_64_v3:
    rustflags = "-C target-cpu=x86-64-v3"
else:
    rustflags = "-C target-cpu=native"

if rustflags:
    rustflags = os.environ.get("RUSTFLAGS", "") + " " + rustflags
    os.environ["RUSTFLAGS"] = rustflags

print(f"Build config: {noflags=} {portable=} {x86_64_v3=} {rustflags=}.")

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
        )
    ],
)
