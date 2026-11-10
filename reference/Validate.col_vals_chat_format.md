# Validate.col_vals_chat_format()

Validate whether column values are well-formed chat conversations.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15598-L15849)

``` python
Validate.col_vals_chat_format(
    columns,
    roles=("system", "user", "assistant"),
    alternate=True,
    first_role=None,
    last_role=None,
    allow_empty=False,
    role_key="role",
    content_key="content",
    na_pass=False,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `col_vals_chat_format()` validation method checks that each value of a column is a well-formed conversation of the kind used to fine-tune a chat model: a list of messages, each with a role and some content. Each row is a test unit, and a row fails if its value isn't a list of messages, has a message without a role or with a role that isn't in `roles=`, has a system message anywhere but first, has turns that don't alternate between roles, starts or ends with the wrong role, or has a message without content. Malformed conversations are common in chat data that's been converted from other formats, and a training run can fail on them, or silently learn from them.

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column. The values are lists of messages (such as a Polars column of lists of structs, or a pandas column of lists of dictionaries) or the JSON text of them, either as a list or as an object with a `"messages"` key.

`roles: Sequence[str] = ("system", "user", "assistant")`  
The roles that a message can have. The default is `("system", "user", "assistant")`. If `"system"` is one of them, a system message can only be the first message.

`alternate: bool = True`  
Should the turns (the messages after a first system message) alternate, so that no role follows itself? The default is `True`.

`first_role: str | None = None`  
The role of the first turn (the first message after a system message), such as `"user"`. The default, `None`, doesn't check it.

`last_role: str | None = None`  
The role of the last message, such as `"assistant"` (for a conversation that ends with the reply that a model should learn). The default, `None`, doesn't check it.

`allow_empty: bool = False`  
Can a message have empty content (missing, or only whitespace)? The default is `False`.

`role_key: str = "role"`  
The key of the role in each message. The default is `"role"`.

`content_key: str = "content"`  
The key of the content in each message. The default is `"content"`.

`na_pass: bool = False`  
Should a missing value pass? The default is `False`.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is `"validity"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Raises

`TypeError`  
If `roles=` isn't a list of role names, or if `role_key=` or `content_key=` isn't a string.

`ValueError`  
If `roles=` is empty, or if `first_role=` or `last_role=` isn't one of `roles=`.

## The Problems Found

The first problem found with a conversation is in a column of the results called `pb_reason_`, which is in the extracts of failing rows and in the step report. The step's note says how many rows have each problem. The problems are checked in this order:

- `not_messages`: the value isn't a list of messages (or the JSON text of one)
- `no_messages`: the list of messages is empty
- `bad_message`: a message isn't a mapping with a role (under `role_key=`) that's a string
- `unknown_role`: a message has a role that isn't in `roles=`
- `system_position`: a system message isn't the first message
- `first_role`: the first turn doesn't have the role of `first_role=`
- `not_alternating`: a role follows itself (with `alternate=True`)
- `last_role`: the last message doesn't have the role of `last_role=`
- `empty_content`: a message has no content, or content that's only whitespace (unless `allow_empty=True`)

A Polars or pandas column can hold the messages as lists (of structs or dictionaries) or as JSON text. For a database table (through Ibis) or a PySpark DataFrame, the column must hold JSON text: its distinct values are collected and checked, and the problems are joined back to the table on its own backend.

## Examples

For the examples here, we'll use a small table of conversations, stored as JSON text.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "messages": [
            '[{"role": "user", "content": "Hi!"}, '
            '{"role": "assistant", "content": "Hello! How can I help?"}]',
            '[{"role": "user", "content": "Hi!"}, {"role": "user", "content": "Hello?"}]',
            '[{"role": "system", "content": "Be brief."}, '
            '{"role": "user", "content": "Capital of France?"}, '
            '{"role": "assistant", "content": "Paris."}]',
            '[{"role": "user", "content": "What is 2 + 2?"}, '
            '{"role": "bot", "content": "4"}]',
        ],
    }
)
```

Let's validate that each conversation is well formed and ends with a reply of the assistant.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_chat_format(columns="messages", last_role="assistant")
    .interrogate()
)

validation
```

[TABLE]

Two of the four conversations fail: the second has two user messages in a row, and the fourth has a message with the role `"bot"`. The step report shows the problem with each failing row in the `pb_reason_` column.

``` python
validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `messages is a chat of {system, user, assistant}` **2** / **4** TEST UNIT FAILURES IN COLUMN **2** EXTRACT OF ALL **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |
|----|----|----|----|
|  | id *Int64* | messages *String* | pb_reason\_ *String* |
| 2 | 2 | \[{"role": "user", "content": "Hi!"}, {"role": "user", "content": "Hello?"}\] | not_alternating |
| 4 | 4 | \[{"role": "user", "content": "What is 2 + 2?"}, {"role": "bot", "content": "4"}\] | unknown_role |

The third conversation passes: its system message is first, and its turns alternate between the user and the assistant.
