# Contributing Guidelines

There are many ways to contribute to the ongoing development of the Pointblank package. Some contributions can be rather easy to do (e.g., fixing typos, improving documentation, filing issues for feature requests or problems, etc.) whereas other contributions can require more time and patience (like answering questions and submitting pull requests with code changes). Just know that help provided in any capacity is very much appreciated.

## Filing Issues

If you believe you found a bug, minimal reproducible example (MRE) for your posting to the [pointblank issue tracker](https://github.com/posit-dev/pointblank/issues). Try not to include anything unnecessary, just the minimal amount of code that constitutes the reproducible bug. For useful guidelines on how to create an MRE, take a look at [this guide on Stack Overflow](https://stackoverflow.com/help/minimal-reproducible-example). We will try to verify the bug by running the code in the provided MRE. The quality of the MRE will reduce the amount of back-and-forth communication in trying to understand how to execute the code on our systems.

## Answering questions

One way to help is by simply answering questions. It's amazing how a little conversation could lead to better insights on a problem. Don't quite know the answer? That's okay too. We're all in this together.

Where might you answer user questions? Some of the forums for Q&A on Pointblank include the _Issues_ and _Discussion_ pages in the repo. Good etiquette is key during these interactions: be a good person to all who ask questions.

### Making Pull Requests

Should you consider making a pull request (PR), please file an issue first and explain the problem in some detail. If the PR is an enhancement, detail how the change would make things better for package users. Bugfix PRs also require some explanation about the bug and how the proposed fix will remove that bug. A great way to illustrate the bug is to include an MRE. While all this upfront work prior to preparing a PR can be time-consuming it opens a line of communication with the package authors and the community, perhaps leading to a better enhancement or more effective fixes!

Once there is consensus that a PR based on the issue would be helpful, adhering to the following process will make things proceed more quickly:

- Create a separate Git branch for each PR
- The Pointblank package follows the [Style Guide for Python Code](https://peps.python.org/pep-0008/) so please adopt those guidelines in your submitted code as best as possible
- Comment your code, particularly in those hard-to-understand areas
- Add test cases that cover the changes made in the PR; having tests for any new codepaths will help guard against regressions

### Setting Up Your Development Environment

To set up your development environment, first clone the posit-dev/pointblank repository.

If you're using UV, you may run `uv sync` and your environment is setup! If using pip or another package manager, keep following these steps:

- Create a virtual environment for the folder.
- Install the package in editable mode with `pip install -e .` from the root of the project folder.
- Install the development dependencies with `pip install '.[dev]'` (have a look at the `pyproject.toml` file for the list of development dependencies)

Our documentation uses `quartodoc` which in turn requires a local install of the Quarto CLI. To install Quarto, go to <https://quarto.org/docs/get-started/> to get the latest build for your platform.

### Building the Documentation Locally

Building the documentation can be done with `make docs-build` from the root of the project folder. Locally building the documentation site is useful when you want to see how your changes will look during iteration. The site will be built in the `docs/_site` folder.

### Running Tests Locally

The tests are located in the `tests` folder and we use `pytest` for running them. To run all of the tests, use `make test`. If you want to run a specific test file, you can use `pytest tests/test_file.py`.

If you create new tests involving snapshots, please ensure that the resulting snapshots are relatively small. After adding snapshots, use `make test-update` (this runs `pytest --snapshot-update`). A subsequent use of `make test` should pass without any issues.

### Creating Aggregation Methods

Aggregation methods are generated dynamically! This is done because they all have the same signature and they're registered on the `Validate` class in the same way. So, to add a new method, go to `pointblank/_agg.py` and add either a comparison or statistical aggregation function.

Comparison functions are defined by `comp_*`, for example `comp_gt` for "greater than". Statistical functions are defined by `agg_*`, for example `agg_sum` for "sum". At build time, these are registered and a grid of all combinations are created:
```{python}
Aggregator = Callable[[nw.DataFrame], Any]
Comparator = Callable[[Any, Any], bool]

AGGREGATOR_REGISTRY: dict[str, Aggregator] = {}

COMPARATOR_REGISTRY: dict[str, Comparator] = {}
```

Once you've added a new method(s), run `make pyi` to generate the updated type stubs in `pointblank/validate.pyi` which contains the new signatures for the aggregation methods. At runtime, or import time to be precise, the methods are added to the `Validate` class and resolved internally through the registry.
```{python}
# pointblank/validate.py
for method in load_validation_method_grid():  # -> `col_sum_*`, `col_mean_*`, etc.
    setattr(Validate, method, make_agg_validator(method))
```

At this point, the methods will exist AND the docs/signature are loaded properly in the type checker and IDE/LSPs, which is very important for usability.
### Linting and Type Checking

We use `ruff` for linting, the settings used are fairly loose and objective. Linting is run in pre-commit in CI. You can run it locally with `make lint`. Type checking is currently not enforced, but we intend on gradually typing the codebase. You can run `make type` to run Astral's new experimental type checker `ty`. Feel free to leverage type hints and occasionally type checking but it's not obligatory at this time.
