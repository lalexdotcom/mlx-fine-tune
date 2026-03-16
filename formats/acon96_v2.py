# formats/acon96_v2.py
#
# Converter for acon96/Home-Assistant-Requests-V2 dataset.
# Format: messages (list of role/content dicts) + tools (list of function defs)
# Training signal: train_on_turn=True on assistant turns to learn from.
#
# This is the V2 format — uses standard OpenAI-compatible tool_calls
# instead of the custom ```homeassistant block from V1.

from pathlib import Path

from formats import MaskedExample, EvalExample, FormatError
from lib.template import render_turns_with_diff, build_prompt


SUPPORTED_ROLES = {"system", "user", "assistant", "tool", "model"}


def _normalize_content(content) -> str:
    """Extract plain text from content, whether it's a string or
    a list of {type, text} blocks as used in the V2 dataset."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return content[0].get("text", "") if content else ""
    return ""


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize raw messages from the dataset into clean dicts
    suitable for Jinja2 chat template rendering."""
    result = []
    for m in messages:
        entry = {
            "role": m["role"],
            "content": _normalize_content(m.get("content", "")),
            "train_on_turn": bool(m.get("train_on_turn", False)),
        }
        if m.get("tool_calls"):
            entry["tool_calls"] = m["tool_calls"]
        result.append(entry)
    return result


def validate(example: dict) -> None:
    """Validate that an example has the expected acon96-v2 structure.
    Raises FormatError with a descriptive message if validation fails."""
    if "messages" not in example:
        raise FormatError("Missing 'messages' field")
    if not isinstance(example["messages"], list):
        raise FormatError("'messages' must be a list")
    for i, msg in enumerate(example["messages"]):
        if "role" not in msg:
            raise FormatError(f"Message {i} missing 'role'")
        if msg["role"] not in SUPPORTED_ROLES:
            raise FormatError(f"Message {i} has unsupported role: {msg['role']}")


def convert_for_training(
    example: dict,
    tokenizer_config: dict,
) -> list[MaskedExample]:
    """Convert one acon96-v2 example into masked training examples.
    Each assistant turn with train_on_turn=True produces one MaskedExample
    where the prompt is everything before that turn and the completion
    is the turn itself.
    Returns an empty list if no trainable turns are found.
    """
    try:
        validate(example)
    except FormatError as e:
        return []

    messages = _normalize_messages(example.get("messages", []))
    tools = example.get("tools") or None

    turns = render_turns_with_diff(tokenizer_config, messages, tools)
    results = []

    for i, (turn_text, should_train) in enumerate(turns):
        if not should_train:
            continue
        prompt = "".join(t for t, _ in turns[:i])
        results.append(MaskedExample(text=prompt + turn_text))

    return results


def convert_for_eval(
    example: dict,
    tokenizer_config: dict,
) -> EvalExample | None:
    """Convert one acon96-v2 example into an evaluation example.
    Uses the last user message as input and the last assistant turn
    as the expected output.
    Returns None if the example cannot be used for evaluation.
    """
    try:
        validate(example)
    except FormatError:
        return None

    messages = _normalize_messages(example.get("messages", []))
    tools = example.get("tools") or None

    # Find last assistant turn to use as expected output
    last_assistant = None
    for msg in reversed(messages):
        if msg["role"] in ("assistant", "model"):
            last_assistant = msg
            break

    if last_assistant is None:
        return None

    # Build prompt from all messages up to (not including) last assistant turn
    last_idx = len(messages) - 1 - next(
        i for i, m in enumerate(reversed(messages))
        if m["role"] in ("assistant", "model")
    )
    prompt_messages = messages[:last_idx]
    prompt = build_prompt(tokenizer_config, prompt_messages, tools)

    # Extract expected tool call if present
    expected_tool = None
    if last_assistant.get("tool_calls"):
        tc = last_assistant["tool_calls"][0]
        fn = tc.get("function", tc)
        expected_tool = {
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", {}),
        }

    return EvalExample(
        prompt=prompt,
        expected_text=last_assistant.get("content", ""),
        expected_tool=expected_tool,
        raw=example,
    )