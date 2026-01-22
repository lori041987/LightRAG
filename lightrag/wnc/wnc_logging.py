"""
WNC custom logging utilities for LightRAG.

This module provides utilities for detailed function tracing according to
wnc_docs/code_analysis_index_260123.txt requirements.
"""

import inspect
import json
import logging
from typing import Any, Literal

# Get WNC-specific logger instance (separate from main lightrag logger)
logger = logging.getLogger("lightrag.wnc")

LogLevel = Literal["trace", "debug", "info", "warning", "error"]
DelimiterType = Literal["pipe", "newline", "comma"]

# Define TRACE log level (lower than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

# Global delimiter setting (can be changed via set_delimiter())
_DELIMITER = " | "


def set_delimiter(delimiter_type: DelimiterType) -> None:
    """
    Set the global delimiter for WNC log messages.

    Args:
        delimiter_type: Type of delimiter to use
            - "pipe": " | " (compact, single line)
            - "newline": "\\n  " (multi-line, easier to read)
            - "comma": ", " (CSV-like)
    """
    global _DELIMITER
    if delimiter_type == "pipe":
        _DELIMITER = " | "
    elif delimiter_type == "newline":
        _DELIMITER = "\n  "
    elif delimiter_type == "comma":
        _DELIMITER = ", "


def get_delimiter() -> str:
    """Get the current delimiter string."""
    return _DELIMITER


def wnc_log(
    *,
    function_name: str | None = None,
    purpose: str,
    inputs: dict[str, Any] | None = None,
    outputs: str | None = None,
    side_effects: str | None = None,
    note: str | None = None,
    level: LogLevel = "info",
    auto_detect_function: bool = True,
) -> None:
    """
    Log a WNC message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__").
                      If None and auto_detect_function=True, will auto-detect from call stack.
        purpose: Brief description of what the function does
        inputs: Optional dict of input parameters and their values
        outputs: Optional description of outputs/return values
        side_effects: Optional description of files/cache touched or state changes
        note: Optional additional context or notes (error handling, edge cases)
        level: Log level (trace, debug, info, warning, error). Default is "info"
        auto_detect_function: If True and function_name is None, auto-detect from call stack

    Example:
        >>> wnc_log(
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"},
        ...     side_effects="Creates working_dir if missing",
        ...     note="Raises if storage backend incompatible",
        ...     level="info"
        ... )
    """
    # Auto-detect function name if not provided
    if function_name is None and auto_detect_function:
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            func_name = caller_frame.f_code.co_name
            # Try to get the class name if it's a method
            if 'self' in caller_frame.f_locals:
                cls_name = caller_frame.f_locals['self'].__class__.__name__
                function_name = f"{cls_name}.{func_name}"
            elif 'cls' in caller_frame.f_locals:
                cls_name = caller_frame.f_locals['cls'].__name__
                function_name = f"{cls_name}.{func_name}"
            else:
                function_name = func_name

    if function_name is None:
        function_name = "unknown"

    message = format_wnc_log(
        function_name=function_name,
        purpose=purpose,
        inputs=inputs,
        outputs=outputs,
        side_effects=side_effects,
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
    side_effects: str | None = None,
    note: str | None = None,
) -> str:
    """
    Create a WNC log message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__")
        purpose: Brief description of what the function does
        inputs: Optional dict of input parameters and their values
        outputs: Optional description of outputs/return values
        side_effects: Optional description of files/cache touched or state changes
        note: Optional additional context or notes (error handling, edge cases)

    Returns:
        Formatted log message string

    Example:
        >>> format_wnc_log(
        ...     function_name="LightRAG.__post_init__",
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"},
        ...     side_effects="Creates working_dir if missing",
        ...     note="Raises if storage backend incompatible"
        ... )
        '[WNC] function=LightRAG.__post_init__ | purpose="Initialize LightRAG instance and configure storages" | inputs=(working_dir=/path/to/dir, llm_model_name=gpt-4) | side_effects="Creates working_dir if missing" | note="Raises if storage backend incompatible"'
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

    if side_effects:
        parts.append(f"side_effects=\"{side_effects}\"")

    if note:
        parts.append(f"note=\"{note}\"")

    return get_delimiter().join(parts)


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
