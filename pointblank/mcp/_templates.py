"""Validation templates for the Pointblank MCP server."""

from typing import Any, Dict, Optional


def get_validation_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get a predefined validation template."""
    return _TEMPLATES.get(template_name)


AVAILABLE_TEMPLATES = [
    "basic_quality",
    "financial_data",
    "customer_data",
    "sensor_data",
    "survey_data",
]

_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "basic_quality": {
        "description": "Basic data quality checks for any dataset",
        "validations": [
            {
                "validation_type": "col_exists",
                "params": {"columns": "id"},
                "description": "Check that ID column exists",
            },
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "id"},
                "description": "ID column should not have null values",
            },
            {
                "validation_type": "rows_distinct",
                "params": {},
                "description": "Check for duplicate rows",
            },
        ],
    },
    "financial_data": {
        "description": "Validation template for financial/transaction data",
        "validations": [
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "amount"},
                "description": "Amount should not be null",
            },
            {
                "validation_type": "col_vals_gt",
                "params": {"columns": "amount", "value": 0},
                "description": "Amount should be positive",
            },
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "date"},
                "description": "Transaction date should not be null",
            },
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "account_id"},
                "description": "Account ID should not be null",
            },
        ],
    },
    "customer_data": {
        "description": "Validation template for customer/user data",
        "validations": [
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "customer_id"},
                "description": "Customer ID should not be null",
            },
            {
                "validation_type": "col_vals_regex",
                "params": {"columns": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
                "description": "Email should be in valid format",
            },
            {
                "validation_type": "col_vals_between",
                "params": {"columns": "age", "left": 0, "right": 120},
                "description": "Age should be between 0 and 120",
            },
            {
                "validation_type": "col_vals_in_set",
                "params": {"columns": "status", "set_": ["active", "inactive", "pending"]},
                "description": "Status should be one of predefined values",
            },
        ],
    },
    "sensor_data": {
        "description": "Validation template for IoT/sensor data",
        "validations": [
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "timestamp"},
                "description": "Timestamp should not be null",
            },
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "sensor_id"},
                "description": "Sensor ID should not be null",
            },
            {
                "validation_type": "col_vals_between",
                "params": {"columns": "temperature", "left": -50, "right": 100},
                "description": "Temperature should be in reasonable range",
            },
            {
                "validation_type": "col_vals_ge",
                "params": {"columns": "humidity", "value": 0},
                "description": "Humidity should be non-negative",
            },
            {
                "validation_type": "col_vals_le",
                "params": {"columns": "humidity", "value": 100},
                "description": "Humidity should not exceed 100%",
            },
        ],
    },
    "survey_data": {
        "description": "Validation template for survey/questionnaire data",
        "validations": [
            {
                "validation_type": "col_vals_not_null",
                "params": {"columns": "response_id"},
                "description": "Response ID should not be null",
            },
            {
                "validation_type": "col_vals_between",
                "params": {"columns": "satisfaction_score", "left": 1, "right": 10},
                "description": "Satisfaction score should be between 1 and 10",
            },
            {
                "validation_type": "col_vals_in_set",
                "params": {
                    "columns": "completion_status",
                    "set_": ["complete", "partial", "abandoned"],
                },
                "description": "Completion status should be one of predefined values",
            },
        ],
    },
}
