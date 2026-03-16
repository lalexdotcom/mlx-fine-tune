# formats/allenporter_msg.py
#
# Converter for allenporter/assist-llm-function-calling-messages dataset.
# Format: standard messages list (same structure as acon96-v2 but without
#         train_on_turn — all assistant turns are considered trainable).
#
# Can be used for both training and evaluation.

from formats import MaskedExample, EvalExample, FormatError
from formats.acon96_v2 import (
    validate,
    _normalize_messages,
    _normalize_content,
)
from lib.template import render_turns_with_diff, build_prompt


def convert_for_training(
    example: dict,
    tokenizer_config: dict,
) -> list[MaskedExample]:
    """Convert one allenporter-msg example into masked training examples.
    Since this format has no train_on_turn field, all assistant turns
    are treated as trainable.
    """
    try:
        validate(example)
    except FormatError:
        return []

    messages = _normalize_messages(example.get("messages", []))
    # Mark all assistant turns as trainable since there is no train_on_turn
    for msg in messages:
        if msg["role"] in ("assistant", "model"):
            msg["train_on_turn"] = True

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
    """Convert one allenporter-msg example into an evaluation example.
    Delegates to acon96_v2 logic since the message format is identical.
    """
    from formats.acon96_v2 import convert_for_eval as acon96_eval
    return acon96_eval(example, tokenizer_config)