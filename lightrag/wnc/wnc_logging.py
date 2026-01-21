"""
WNC custom logging utilities for LightRAG.

This module provides utilities for detailed function tracing according to
wnc_docs/code_analysis_index_260123.txt requirements.
"""

import logging
from typing import Any, Literal

# Get logger instance
logger = logging.getLogger("lightrag")

LogLevel = Literal["trace", "debug", "info", "warning", "error"]

# Define TRACE log level (lower than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def wnc_log(
    *,
    function_name: str,
    purpose: str,
    inputs: dict[str, Any] | None = None,
    outputs: str | None = None,
    note: str | None = None,
    level: LogLevel = "info",
) -> None:
    """
    Log a WNC message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__")
        purpose: Brief description of what the function does
        inputs: Optional dict of input parameters and their values
        outputs: Optional description of outputs/return values
        note: Optional additional context or notes
        level: Log level (trace, debug, info, warning, error). Default is "info"

    Example:
        >>> wnc_log(
        ...     function_name="LightRAG.__post_init__",
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"},
        ...     note="Creates working_dir if missing",
        ...     level="info"
        ... )
    """
    message = format_wnc_log(
        function_name=function_name,
        purpose=purpose,
        inputs=inputs,
        outputs=outputs,
        note=note,
    )

    # Log at the specified level
    if level == "trace":
        logger.log(TRACE_LEVEL, message)
    else:
        log_method = getattr(logger, level)
        log_method(message)


def format_wnc_log(
    *,
    function_name: str,
    purpose: str,
    inputs: dict[str, Any] | None = None,
    outputs: str | None = None,
    note: str | None = None,
) -> str:
    """
    Create a WNC log message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__")
        purpose: Brief description of what the function does
        inputs: Optional dict of input parameters and their values
        outputs: Optional description of outputs/return values
        note: Optional additional context or notes

    Returns:
        Formatted log message string

    Example:
        >>> format_wnc_log(
        ...     function_name="LightRAG.__post_init__",
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"},
        ...     note="Creates working_dir if missing"
        ... )
        '[WNC] function=LightRAG.__post_init__ | purpose="Initialize LightRAG instance and configure storages" | inputs=(working_dir=/path/to/dir, llm_model_name=gpt-4) | note="Creates working_dir if missing"'
    """
    parts = [
        f"[WNC] function={function_name}",
        f"purpose=\"{purpose}\"",
    ]

    if inputs:
        # Format inputs as key=value pairs, truncating long values
        input_strs = []
        for key, value in inputs.items():
            value_str = _format_value(value)
            input_strs.append(f"{key}={value_str}")
        parts.append(f"inputs=({', '.join(input_strs)})")

    if outputs:
        parts.append(f"outputs=\"{outputs}\"")

    if note:
        parts.append(f"note=\"{note}\"")

    return " | ".join(parts)


def _format_value(value: Any, max_length: int = 100) -> str:
    """
    Format a value for logging, with truncation for long values.

    Args:
        value: The value to format
        max_length: Maximum length before truncation

    Returns:
        Formatted string representation of the value
    """
    # Handle None
    if value is None:
        return "None"

    # Handle lists - show count if too long
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        elif len(value) <= 3:
            items = [_format_value(item, max_length // 3) for item in value]
            return f"[{', '.join(items)}]"
        else:
            return f"[{len(value)} items]"

    # Handle dicts - show count if too long
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        elif len(value) <= 3:
            items = [f"{k}={_format_value(v, max_length // 3)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return f"{{{len(value)} items}}"

    # Convert to string and truncate if needed
    value_str = str(value)
    if len(value_str) > max_length:
        return value_str[:max_length] + "..."

    return value_str
