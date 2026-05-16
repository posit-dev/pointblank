# pb make-template


Create a validation script or YAML configuration template.


``` bash
pb make-template [OUTPUT_FILE]
```


- Creates a sample Python script or YAML configuration with examples showing how to use Pointblank
- for data validation. The template type is determined by the file extension:
- .py files create Python script templates
- .yaml/.yml files create YAML configuration templates

Edit the template to add your own data loading and validation rules, then run it with `pb run`.

OUTPUT_FILE is the path where the template will be created.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb make-template [OPTIONS] [OUTPUT_FILE]

      Create a validation script or YAML configuration template.

      Creates a sample Python script or YAML configuration with examples showing
      how to use Pointblank for data validation. The template type is determined
      by the file extension: - .py files create Python script templates -
      .yaml/.yml files create YAML configuration templates

      Edit the template to add your own data loading and validation rules, then
      run it with 'pb run'.

      OUTPUT_FILE is the path where the template will be created.

      Examples:

      pb make-template my_validation.py        # Creates Python script template
      pb make-template my_validation.yaml      # Creates YAML config template
      pb make-template validation_template.yml # Creates YAML config template

    Options:
      --help  Show this message and exit.


# Arguments


`OUTPUT_FILE: PATH`  
Optional.


# Examples

``` bash

pb make-template my_validation.py        # Creates Python script template
pb make-template my_validation.yaml      # Creates YAML config template
pb make-template validation_template.yml # Creates YAML config template
```
