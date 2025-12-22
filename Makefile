.PHONY: check

.PHONY: pyi
pyi: ## Generate .pyi stub files
	@uv run stubgen ./pointblank/validate.py \
		--include-private \
		-o  .
	@uv run scripts/generate_agg_validate_pyi.py

.PHONY: test
test:
	@uv run pytest tests \
		--cov=pointblank \
		--cov-report=term-missing \
		--randomly-seed 123 \
		-n auto \
		--reruns 3 \
		--reruns-delay 1 \
		--doctest-modules pointblank \
		--durations 10

.PHONY: test-core
test-core: ## Run core libraries only; useful for local CI
	@SKIP_PYSPARK_TESTS=1 \
		SKIP_SQLITE_TESTS=1 \
		SKIP_PARQUET_TESTS=1 \
		uv run pytest \
		--cov=pointblank \
		--cov-report=term-missing \
		--randomly-seed 123 \
		-n auto \
		--durations=10


test-update:
	pytest --snapshot-update

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks
	@uvx pre-commit run --all-files

.PHONY: lint
lint: ## Run ruff formatter and linter
	@uv run ruff format
	@uv run ruff check --fix

.PHONY: install-pre-commit
install-pre-commit: # Install pre-commit hooks
	@uvx pre-commit install

.PHONY: run-pre-commit
run-pre-commit: # Run pre-commit hooks
	@uvx pre-commit run --all-files


type: ## Run experimental type checking
	@uv run ty check pointblank


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

docs-pdf: ## Build PDF version of User Guide (HTML to PDF preserving graphics)
	@echo "Preparing PDF document (stripping YAML from includes)..."
	uv run python scripts/create_pdf_doc.py
	@echo "Rendering User Guide to self-contained HTML..."
	cd docs && uv run quarto render user-guide-pdf-clean.qmd --to html --output user-guide-pdf.html
	@echo "Converting HTML to PDF with Chrome (preserves validation reports)..."
	uv run python scripts/html_to_pdf.py docs/_site/user-guide-pdf.html docs/user-guide.pdf
	@echo "Creating Table of Contents page with actual page numbers..."
	uv run python scripts/create_toc_pdf.py docs/user-guide.pdf
	@echo "PDF available at docs/user-guide.pdf"

docs-llms: ## Generate llms.txt and llms-full.txt files for LLM consumption
	@uv run python scripts/generate_llms_txt.py

docs-full: docs-build docs-llms ## Build docs and generate llms.txt files

install: dist ## install the package to the active Python's site-packages
	python3 -m pip install --force-reinstall dist/pointblank*.whl
