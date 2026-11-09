"""Remove only a single whole-response Markdown fence, uniformly for all models."""
import re

PATTERN = re.compile(r'[ \t\r\n]*```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```[ \t\r\n]*', re.IGNORECASE)


def unwrap(content):
    if not isinstance(content, str):
        return content, False
    match = PATTERN.fullmatch(content)
    if match is None or re.search(r'(?m)^[ \t]*```', match.group(1)):
        return content, False
    return match.group(1), True
