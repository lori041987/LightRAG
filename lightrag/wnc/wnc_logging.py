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

# Global trace content limit (can be changed via set_trace_content_limit())
_TRACE_CONTENT_LIMIT = 10000  # 0 = unlimited

# JSON formatting settings for inputs/outputs
JSON_ENSURE_ASCII = False
JSON_INDENT = 2
JSON_UNESCAPE_FOR_READABILITY = True  # Unescape \n and \" for better readability in logs
JSON_AUTO_SERIALIZE_NONSTANDARD = True  # Auto-convert non-JSON-serializable types (tuples, sets, etc.)


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


def set_trace_content_limit(limit: int) -> None:
    """
    Set the global content limit for trace-level logs.

    Args:
        limit: Maximum characters to show for content values in trace logs.
               0 = unlimited, positive int = max length
    """
    global _TRACE_CONTENT_LIMIT
    _TRACE_CONTENT_LIMIT = limit


def get_trace_content_limit() -> int:
    """Get the current trace content limit."""
    return _TRACE_CONTENT_LIMIT


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects to JSON-serializable formats.

    Handles:
    - Tuples -> Lists
    - Sets -> Lists
    - Callables/Functions -> Function name string
    - Dictionaries with tuple keys -> Dictionaries with string keys (tuple converted to string)
    - Other types -> String representation

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    # Handle None, bool, int, float, str (already JSON-serializable)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Handle callables/functions - extract function name
    if callable(obj):
        # Try to get the function name
        if hasattr(obj, '__name__'):
            return f"<function {obj.__name__}>"
        elif hasattr(obj, 'func') and hasattr(obj.func, '__name__'):
            # For partial or wrapped functions
            return f"<function {obj.func.__name__}>"
        else:
            return "<callable>"

    # Convert tuples to lists
    if isinstance(obj, tuple):
        return [_make_json_serializable(item) for item in obj]

    # Convert sets to lists
    if isinstance(obj, set):
        return [_make_json_serializable(item) for item in obj]

    # Handle lists
    if isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]

    # Handle dictionaries (including those with tuple keys)
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Convert tuple/non-string keys to strings
            if isinstance(key, tuple):
                # Convert tuple to a readable string format like "[src, rel, tgt]"
                serializable_key = str(list(key))
            elif not isinstance(key, (str, int, float, bool, type(None))):
                serializable_key = str(key)
            else:
                serializable_key = key

            result[serializable_key] = _make_json_serializable(value)
        return result

    # For other types, convert to string representation
    return str(obj)


def wnc_log(
    *,
    function_name: str | None = None,
    purpose: str,
    inputs: dict[str, Any] | None = None,
    outputs: str | dict[str, Any] | None = None,
    side_effects: str | None = None,
    note: str | None = None,
    level: LogLevel = "info",
    auto_detect_function: bool = True,
    _is_trace: bool | None = None,
) -> None:
    """
    Log a WNC message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__").
                      If None and auto_detect_function=True, will auto-detect from call stack.
        purpose: Brief description of what the function does
        inputs: Optional dict of input parameters and their values
        outputs: Optional description of outputs/return values (dict will be auto-formatted as JSON)
        side_effects: Optional description of files/cache touched or state changes
        note: Optional additional context or notes (error handling, edge cases)
        level: Log level (trace, debug, info, warning, error). Default is "info"
        auto_detect_function: If True and function_name is None, auto-detect from call stack

    Example:
        >>> wnc_log(
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"},
        ...     outputs={"status": "initialized"},
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

    # Determine if this is trace level for content formatting
    is_trace = _is_trace if _is_trace is not None else (level == "trace")

    # Auto-format inputs if it's a dict
    formatted_inputs = inputs
    if isinstance(inputs, dict):
        # Make inputs JSON-serializable before dumping if enabled
        if JSON_AUTO_SERIALIZE_NONSTANDARD:
            serializable_inputs = _make_json_serializable(inputs)
        else:
            serializable_inputs = inputs
        formatted_inputs = json.dumps(serializable_inputs, ensure_ascii=JSON_ENSURE_ASCII, indent=JSON_INDENT)
        # Unescape common escape sequences for better readability in logs
        if JSON_UNESCAPE_FOR_READABILITY:
            formatted_inputs = formatted_inputs.replace('\\n', '\n').replace('\\"', '"')

    # Auto-format outputs if it's a dict
    formatted_outputs = outputs
    if isinstance(outputs, dict):
        # Make outputs JSON-serializable before dumping if enabled
        if JSON_AUTO_SERIALIZE_NONSTANDARD:
            serializable_outputs = _make_json_serializable(outputs)
        else:
            serializable_outputs = outputs
        formatted_outputs = json.dumps(serializable_outputs, ensure_ascii=JSON_ENSURE_ASCII, indent=JSON_INDENT)
        # Unescape common escape sequences for better readability in logs
        if JSON_UNESCAPE_FOR_READABILITY:
            formatted_outputs = formatted_outputs.replace('\\n', '\n').replace('\\"', '"')

    message = format_wnc_log(
        function_name=function_name,
        purpose=purpose,
        inputs=formatted_inputs,
        outputs=formatted_outputs,
        side_effects=side_effects,
        note=note,
        is_trace=is_trace,
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
    inputs: str | None = None,
    outputs: str | None = None,
    side_effects: str | None = None,
    note: str | None = None,
    is_trace: bool = False,
) -> str:
    """
    Create a WNC log message for function entry/exit with purpose and I/O information.

    Args:
        function_name: Name of the function or method (e.g., "LightRAG.__post_init__")
        purpose: Brief description of what the function does
        inputs: Optional pre-formatted JSON string of input parameters
        outputs: Optional pre-formatted JSON string or description of outputs/return values
        side_effects: Optional description of files/cache touched or state changes
        note: Optional additional context or notes (error handling, edge cases)

    Returns:
        Formatted log message string

    Example:
        >>> format_wnc_log(
        ...     function_name="LightRAG.__post_init__",
        ...     purpose="Initialize LightRAG instance and configure storages",
        ...     inputs='{"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"}',
        ...     side_effects="Creates working_dir if missing",
        ...     note="Raises if storage backend incompatible"
        ... )
        '[WNC] function=LightRAG.__post_init__ | purpose="Initialize LightRAG instance and configure storages" | inputs={"working_dir": "/path/to/dir", "llm_model_name": "gpt-4"} | side_effects="Creates working_dir if missing" | note="Raises if storage backend incompatible"'
    """
    parts = [
        f"[WNC] function={function_name}",
        f"purpose=\"{purpose}\"",
    ]

    if inputs:
        parts.append(f"inputs={inputs}")

    if outputs:
        parts.append(f"outputs={outputs}")

    if side_effects:
        # Add newline at the beginning for better readability
        parts.append(f"side_effects=\"\n{side_effects}\"")

    if note:
        # Add newline at the beginning for better readability
        parts.append(f"note=\"\n{note}\"")

    return get_delimiter().join(parts)


def _format_value(value: Any, max_length: int = 100, is_trace: bool = False) -> str:
    """
    Format a value for logging, with truncation for long values.

    Args:
        value: The value to format
        max_length: Maximum length before truncation (ignored if is_trace=True)
        is_trace: If True, use trace content limit instead of max_length

    Returns:
        Formatted string representation of the value
    """
    # Use trace content limit for trace-level logs
    if is_trace:
        trace_limit = get_trace_content_limit()
        if trace_limit > 0:
            max_length = trace_limit
        else:
            max_length = float('inf')  # Unlimited
    # Handle None
    if value is None:
        return "None"

    # Handle lists - show count if too long
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        elif len(value) <= 3:
            items = [_format_value(item, max_length // 3, is_trace) for item in value]
            return f"[{', '.join(items)}]"
        else:
            return f"[{len(value)} items]"

    # Handle dicts - show count if too long
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        elif len(value) <= 3:
            items = [f"{k}={_format_value(v, max_length // 3, is_trace)}" for k, v in value.items()]
            return f"{{{', '.join(items)}}}"
        else:
            return f"{{{len(value)} items}}"

    # Convert to string and truncate if needed
    value_str = str(value)
    if len(value_str) > max_length:
        return value_str[:max_length] + "..."

    return value_str
