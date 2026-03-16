# formats/__init__.py

from dataclasses import dataclass


@dataclass
class MaskedExample:
    """A single training example with prompt masked from loss calculation.
    The full text contains both prompt and completion — mlx_lm computes
    loss only on tokens that follow the prompt portion.
    """
    text: str


@dataclass
class EvalExample:
    """A single evaluation example ready to be sent to the model.
    Contains the formatted prompt, expected outputs, and the raw source
    row for inclusion in the evaluation report.
    """
    prompt: str                  # formatted prompt ready to send to the model
    expected_text: str           # expected plain text response (may be empty)
    expected_tool: dict | None   # expected tool call {"name": ..., "arguments": ...}
    raw: dict                    # original dataset row for reporting


class FormatError(Exception):
    """Raised when an example cannot be parsed with the given format."""
    pass