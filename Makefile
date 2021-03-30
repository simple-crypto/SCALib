
.PHONY: pipenv dev test devtest docs

pipenv:
	pipenv sync

dev:
	pipenv run python setup.py develop

test:
	pipenv run pytest

devtest: dev test

docs: dev
	cd docs && pipenv run make html
