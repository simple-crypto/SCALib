.PHONY: dev test devtest coverage docs codestyle fmt wheel help

help:
	@echo "Commands for SCALib development:"
	@echo "    dev: Build in debug mode and run tests."
	@echo "         The associated virtualenv in as .tox/dev, and the SCALib install"
	@echo "         is editable: changes in the python source code are automatically used."
	@echo "         (You'll still need to re-run if you update rust code.)"
	@echo "    test: Build in release mode and run tests, also checks formatting."
	@echo "    test_mindeps: Build in release mode and run tests using minimum version of python dependencies."
	@echo "    coverage: Like test, but also generates detailed coverage report."
	@echo "    docs: Generate docs."
	@echo "    codestyle: Check fromatting."
	@echo "    fmt: Format code."
	@echo "    wheel_local: Build a wheel for the local machine."
	@echo "    wheel_portable: Build a wheel for with maximal portability (at the expense of efficiency)."
	@echo "    wheel_x86_64_v3: Build a wheel for X86_64 v3 level CPUs."

dev:
	tox run -e dev

test:
	tox run -e fmt -- --check
	RUST_BACKTRACE=1 CARGO_TARGET_DIR=.cargo_build cargo test --workspace --manifest-path src/scalib_ext/Cargo.toml
	tox run -e test

test_mindeps:
	tox run -e min_deps

coverage:
	tox run -e test
	tox run -e coverage
	@echo "Open htmlcov/index.html to see detailled coverage information."

docs:
	tox run -e docs

codestyle:
	tox run -e fmt -- --check

fmt:
	tox run -e fmt

wheel_x86_64_v3:
	SCALIB_X86_64_V3=1 CARGO_TARGET_DIR=.cargo_build_x86_64_v3 pyproject-build -m build -w -o dist/avx2

wheel_local:
	CARGO_TARGET_DIR=.cargo_build pyproject-build -w -o dist/local

wheel_portable:
	SCALIB_PORTABLE=1 CARGO_TARGET_DIR=.cargo_build_portable  pyproject-build -w -o dist/portable
