from typing import List


def build_context(
    messages: List[str], max_messages: int = 5, separator: str = " "
) -> str:
    """
    Concatenate multiple chat bubbles into a single context string.
    Keeps recent messages only.
    """

    if not messages:
        return ""

    recent = messages[-max_messages:]
    cleaned = [m.strip() for m in recent if m.strip()]
    return " ".join(cleaned)
