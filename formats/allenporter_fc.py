# formats/allenporter_fc.py
#
# Converter for allenporter/assist-llm-function-calling dataset.
# Format: instructions (system prompt str), tools (list), input (user str),
#         output (expected text response), tool_calls (expected tool call or null)
#
# This dataset is human-validated and intended for evaluation, not training.
# convert_for_training() is intentionally not supported.

from formats import MaskedExample, EvalExample, FormatError
from lib.template import build_prompt


def validate(example: dict) -> None:
    """Validate that an example has the expected allenporter-fc structure.
    Raises FormatError with a descriptive message if validation fails."""
    for field in ["instructions", "tools", "input"]:
        if field not in example:
            raise FormatError(f"Missing required field: '{field}'")
    if not isinstance(example["tools"], list):
        raise FormatError("'tools' must be a list")
    if not isinstance(example["input"], str):
        raise FormatError("'input' must be a string")


def convert_for_training(
    example: dict,
    tokenizer_config: dict,
) -> list[MaskedExample]:
    """Not supported for this format — allenporter-fc is evaluation-only.
    Always returns an empty list."""
    return []


def convert_for_eval(
    example: dict,
    tokenizer_config: dict,
) -> EvalExample | None:
    """Convert one allenporter-fc example into an evaluation example.
    Builds a prompt from the system instructions, tools, and user input
    using the model's chat template.
    Returns None if the example cannot be parsed.
    """
    try:
        validate(example)
    except FormatError as e:
        print(f"  ⚠ Skipping invalid example: {e}")
        return None

    # Build standard messages list from flat fields
    messages = [
        {"role": "system", "content": example["instructions"]},
        {"role": "user",   "content": example["input"]},
    ]

    tools = example.get("tools") or None

    prompt = build_prompt(tokenizer_config, messages, tools)

    # Extract expected tool call if present
    expected_tool = None
    raw_tool_calls = example.get("tool_calls")
    if raw_tool_calls:
        tc = raw_tool_calls[0] if isinstance(raw_tool_calls, list) else raw_tool_calls
        fn = tc.get("function", tc)
        expected_tool = {
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", {}),
        }

    return EvalExample(
        prompt=prompt,
        expected_text=example.get("output", ""),
        expected_tool=expected_tool,
        raw=example,
    )