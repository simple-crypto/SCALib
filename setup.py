from setuptools import setup
from setuptools_rust import Binding, RustExtension

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scale",
    version="0.1.0",
    author="Olivier Bronchain, Gaëtan Cassiers",
    author_email="olivier.bronchain@uclouvain.be gaetan.cassiers@uclouvain.be",
    description="Side-Channel Attacks and Leakage Evaluation",
    long_description=long_description,
    long_description_content_type="text/rst",
    url="https://github.com/obronchain/scale",
    project_urls={
        "Bug Tracker": "https://github.com/obronchain/scale/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    rust_extensions=[
        RustExtension(
            "scale._scale_ext",
            path="scale_ext/scale-py/Cargo.toml",
            binding=Binding.PyO3,
            features=["pyo3/abi3"],
            py_limited_api=True,
        )
    ],
    packages=["scale"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "numpy~=1.19.1",
        "tqdm~=4.38",
        "scipy~=1.5.2",
        "scikit-learn~=0.23.1",
    ],
)
