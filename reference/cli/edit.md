# pb edit


Edit an existing validation plan with a natural-language instruction (AI Validation Editor).


``` bash
pb edit [OPTIONS] [VALIDATION_FILE]
```


VALIDATION_FILE is a Python (.py) or YAML (.yaml/.yml) file containing a validation plan. The instruction is sent, along with the current plan, to the specified model; the proposed change is shown as a diff for review and can optionally be written to a file.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb edit [OPTIONS] [VALIDATION_FILE]

      Edit an existing validation plan with a natural-language instruction (AI
      Validation Editor).

      VALIDATION_FILE is a Python (.py) or YAML (.yaml/.yml) file containing a
      validation plan. The instruction is sent, along with the current plan, to
      the specified model; the proposed change is shown as a diff for review and
      can optionally be written to a file.

      Examples:

      pb edit plan.py -i "add a not-null check on user_id" -m anthropic:claude-opus-4-8
      pb edit plan.yaml -i "tighten the price range to 0-1000" -m openai:gpt-4o --data sales.csv
      pb edit plan.py -i "drop the email regex check" -m anthropic:claude-opus-4-8 -o plan2.py -y

    Options:
      -i, --instruction TEXT  Plain-English description of the change to make to
                              the plan.
      -m, --model TEXT        Model to use, as provider:model (e.g.,
                              anthropic:claude-opus-4-8).
      --data TEXT             Optional data source used for DataScan-informed
                              edits and validation.
      -o, --output PATH       Write the revised plan to this file.
      -y, --yes               Write the output file without asking for
                              confirmation.
      --help                  Show this message and exit.


# Arguments


`VALIDATION_FILE: PATH`  
Optional.


# Options


`-i, --instruction: TEXT`  
Plain-English description of the change to make to the plan.

`-m, --model: TEXT`  
Model to use, as provider:model (e.g., anthropic:claude`-o`pus-4-8).

`--data: TEXT`  
Optional data source used for DataScan`-i`nformed edits and validation.

`-o, --output: PATH`  
Write the revised plan to this file.

`-y, --yes`  
Write the output file without asking for confirmation.


# Examples

``` bash

pb edit plan.py -i "add a not-null check on user_id" -m anthropic:claude-opus-4-8
pb edit plan.yaml -i "tighten the price range to 0-1000" -m openai:gpt-4o --data sales.csv
pb edit plan.py -i "drop the email regex check" -m anthropic:claude-opus-4-8 -o plan2.py -y
```
