# pointblank-example-checks

An example of a package that provides check collections and a thresholds preset to
[Pointblank](https://github.com/posit-dev/pointblank). Once it's installed, a composed
Pointblank config can select what it provides by name:

```yaml
defaults:
  - override /pointblank/thresholds: example/no_failures
  - override /pointblank/checks:
      - example/ids
      - example/emails
  - _self_

vars:
  id_column: customer_id
  email_column: email

tbl: customers.csv
```

The package has no Python code of its own. It has a config tree in
`pointblank_example_checks/conf/` and an entry point in `pyproject.toml` that puts that tree on
the Hydra search path.

The files are under a directory named after the provider (`example/`), and each check
collection has a `# @package checks.<name>` header with a name that no other collection is
likely to use. Both keep one package's options from replacing another's.
