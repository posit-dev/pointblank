.PHONY: check

.PHONY: test
test:
	@uv run pytest \
		--cov=pointblank \
		--cov-report=term-missing \
		--randomly-seed 123 \
		-n auto \
		--reruns 3 \
		--reruns-delay 1

test-update:
	pytest --snapshot-update


lint: ## Run ruff formatter and linter
	@uv run ruff format
	@uv run ruff check --fix

check:
	pyright --pythonversion 3.8 pointblank
	pyright --pythonversion 3.9 pointblank
	pyright --pythonversion 3.10 pointblank
	pyright --pythonversion 3.11 pointblank

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel
	ls -l dist

docs-build:
	cd docs \
	  && quartodoc build --verbose \
	  && quarto render

install: dist ## install the package to the active Python's site-packages
	python3 -m pip install --force-reinstall dist/pointblank*.whl
