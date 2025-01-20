import os
import sys

from setuptools import setup
from setuptools_rust import Binding, RustExtension

# Ensure these are present (in case we are not using PEP-518 compatible build
# system).
import setuptools_scm


# BEGIN RUSTFLAGS hack
from setuptools_rust import build

old_prepare_build_environment = build._prepare_build_environment


def new_prepare_build_environment():
    import inspect

    env = old_prepare_build_environment()
    ext = inspect.stack()[1][0].f_locals["ext"]
    try:
        env["RUSTFLAGS"] = " ".join([env.get("RUSTFLAGS", ""), ext.more_rustflags])
    except AttributeError:
        pass
    print("injected RUSTFLAGS", env.get("RUSTFLAGS"))
    return env


build._prepare_build_environment = new_prepare_build_environment


class RustExtensionFlags(RustExtension):
    def __init__(self, *args, more_rustflags, **kwargs):
        super(RustExtensionFlags, self).__init__(*args, **kwargs)
        self.more_rustflags = more_rustflags


# END RUSTFLAGS hack


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

# if rustflags:
#    rustflags = os.environ.get("RUSTFLAGS", "") + " " + rustflags
#    os.environ["RUSTFLAGS"] = rustflags

print(f"Build config: {noflags=} {portable=} {x86_64_v3=} {rustflags=}.")

scalib_features = ["pyo3/abi3"]

if env_true("SCALIB_BLIS"):
    scalib_features.append("blis")

setup(
    project_urls={
        "Bug Tracker": "https://github.com/simple-crypto/scalib/issues",
    },
    rust_extensions=[
        RustExtensionFlags(
            "scalib._scalib_ext",
            path="src/scalib_ext/scalib-py/Cargo.toml",
            binding=Binding.PyO3,
            features=scalib_features,
            py_limited_api=True,
            rust_version=">=1.83",
            more_rustflags=rustflags,
        ),
        RustExtension(
            "scalib._cpu_check",
            path="src/scalib_ext/cpu-check/Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
            rust_version=">=1.83",
        ),
    ],
)
