.PHONY: check

.PHONY: test
test:
	@pytest \
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

.PHONY: benchmark benchmark-small benchmark-medium benchmark-large benchmark-xlarge benchmark-generate-data

# Default benchmark with small dataset
benchmark: benchmark-small

# Generate benchmark datasets
benchmark-generate-data:
	@echo "Generating benchmark datasets..."
	@mkdir -p benchmark/data
	@python benchmark/generate_data.py

# Small benchmark dataset generation (1K, 10K rows)
benchmark-generate-small:
	@echo "Generating small benchmark datasets..."
	@mkdir -p benchmark/data
	@python benchmark/generate_data.py --sizes 1000 10000 --columns 10

# Medium benchmark dataset generation (10K, 100K rows)
benchmark-generate-medium:
	@echo "Generating medium benchmark datasets..."
	@mkdir -p benchmark/data
	@python benchmark/generate_data.py --sizes 10000 100000 --columns 10 20

# Large benchmark dataset generation (100K, 1M rows)
benchmark-generate-large:
	@echo "Generating large benchmark datasets..."
	@mkdir -p benchmark/data
	@python benchmark/generate_data.py --sizes 100000 1000000 --columns 10 20 50

# Extra large benchmark dataset generation (1M, 10M rows)
benchmark-generate-xlarge:
	@echo "Generating extra large benchmark datasets..."
	@mkdir -p benchmark/data
	@python benchmark/generate_data.py --sizes 1000000 10000000 --columns 10 20

# Quick benchmark with small datasets
benchmark-small:
	@echo "Running small benchmark..."
	@mkdir -p benchmark/results
	@python benchmark/benchmark.py --config small

# Medium-sized benchmark
benchmark-medium:
	@echo "Running medium benchmark..."
	@mkdir -p benchmark/results
	@python benchmark/benchmark.py --config medium

# Large dataset benchmark
benchmark-large:
	@echo "Running large benchmark suite (this may take a while)..."
	@mkdir -p benchmark/results
	@python benchmark/benchmark.py --config large

# Extra large dataset benchmark
benchmark-xlarge:
	@echo "Running extra large benchmark suite (this will take a long time)..."
	@mkdir -p benchmark/results
	@python benchmark/benchmark.py --config xlarge

benchmark-report:
	@echo "Generating benchmark report..."
	@mkdir -p benchmark/reports
	@python benchmark/report.py
