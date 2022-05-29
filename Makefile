.PHONY: dev test devtest coverage docs codestyle fmt wheel


dev:
	tox -e dev
	@echo "Run source .tox/dev/bin/activate to activate development virtualenv."

test:
	tox -e test

devtest:
	tox -e dev pytest

coverage:
	tox -e test-cov
	@echo "Open htmlcov/index.html to see detailled coverage information."

docs:
	tox -e build_docs

codestyle:
	tox -e codestyle

fmt:
	tox -e fmt

wheel:
	RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=.cargo_build pip wheel . --no-deps
