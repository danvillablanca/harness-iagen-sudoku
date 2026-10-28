# response_parser.py

import json
import re
from dataclasses import dataclass


JSON_WHITESPACE = " \t\r\n"
WS = r"[ \t\r\n]*"

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FENCE_RE = re.compile(
    r"```(?:json)?[ \t\r\n]*(.*?)[ \t\r\n]*```",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedAnswer:
    grid: list[list[int]] | None
    source: str
    format_ok: bool
    issues: tuple[str, ...]


def parse_answer(response: str, size: int) -> ParsedAnswer:
    if type(size) is not int or size < 1:
        raise ValueError("size must be a positive integer.")

    # Remove a completed reasoning section when thinking mode is enabled.
    text = response.rsplit("</think>", 1)[-1].strip(JSON_WHITESPACE)

    issues = []
    format_ok = False

    markers = list(re.finditer(r"</?answer\b", text, re.IGNORECASE))

    if markers:
        matches = list(ANSWER_RE.finditer(text))

        if (
            len(markers) != 2
            or len(matches) != 1
            or text.count("<answer>") != 1
            or text.count("</answer>") != 1
        ):
            return ParsedAnswer(
                grid=None,
                source="answer_tags",
                format_ok=False,
                issues=(
                    "Expected exactly one complete, correctly spelled "
                    "<answer> block.",
                ),
            )

        match = matches[0]
        payload = match.group(1).strip(JSON_WHITESPACE)
        source = "answer_tags"

        format_ok = text == match.group(0)

        if not format_ok:
            issues.append("Extra text appears outside the answer block.")

    else:
        # Recover the array for diagnostics without accepting the wrong format.
        fence = FENCE_RE.fullmatch(text)

        if fence is not None:
            payload = fence.group(1).strip(JSON_WHITESPACE)
            source = "markdown_json"
        else:
            payload = text
            source = "bare_json"

        issues.append(
            "Required <answer> tags are missing; parsed for diagnostics only."
        )

    # Permit zero and other nonnegative integers during diagnostic parsing.
    # Sudoku value restrictions are checked separately.
    integer = r"(?:0|[1-9][0-9]*)"

    row_pattern = (
        rf"\[{WS}{integer}"
        rf"(?:{WS},{WS}{integer}){{{size - 1}}}{WS}\]"
    )
    grid_pattern = (
        rf"\[{WS}{row_pattern}"
        rf"(?:{WS},{WS}{row_pattern}){{{size - 1}}}{WS}\]"
    )

    if re.fullmatch(grid_pattern, payload) is None:
        issues.append(
            f"Expected a {size}x{size} JSON array containing "
            "nonnegative integers."
        )
        return ParsedAnswer(None, source, False, tuple(issues))

    try:
        grid = json.loads(payload)
    except ValueError:
        issues.append("The array could not be decoded as JSON.")
        return ParsedAnswer(None, source, False, tuple(issues))

    return ParsedAnswer(grid, source, format_ok, tuple(issues))