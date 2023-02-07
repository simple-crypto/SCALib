.PHONY: dev test devtest coverage docs codestyle fmt wheel help

help:
	@echo "Commands for SCALib development:"
	@echo "    dev: Build in debug mode and run tests."
	@echo "         The associated virtualenv in as .tox/dev, and the SCALib install"
	@echo "         is editable: changes in the python source code are automatically used."
	@echo "         (You'll still need to re-run if you update rust code.)"
	@echo "    test: Build in release mode and run tests, also checks formatting."
	@echo "    coverage: Like test, but also generates detailed coverage report."
	@echo "    docs: Generate docs."
	@echo "    codestyle: Check fromatting."
	@echo "    fmt: Format code."
	@echo "    wheel_local: Build a wheel for the local machine."
	@echo "    wheel_portable: Build a wheel for with maximal portability (at the expense of efficiency)."
	@echo "    wheel_avx2: Build a wheel with AVX2 instruction (should work on any recent x86-64)."

dev:
	tox run -e dev

test:
	tox run -e fmt -- --check
	tox run -e test

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

wheel_avx2:
	SCALIB_AVX2=1 CARGO_TARGET_DIR=.cargo_build_avx2 pip wheel . --no-deps

wheel_local:
	CARGO_TARGET_DIR=.cargo_build pip wheel . --no-deps

wheel_portable:
	SCALIB_PORTABLE=1 CARGO_TARGET_DIR=.cargo_build_portable pip wheel . --no-deps

