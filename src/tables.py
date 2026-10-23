"""Write dataframes as Markdown tables.

Written out rather than taken from a library so that the repository runs on a
bare pandas installation. Columns of floating-point numbers are formatted to a
fixed number of decimals, which keeps the committed tables stable from one run
to the next and makes the diffs readable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def format_frame(frame: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Return a copy whose cells are all strings, ready to be laid out."""
    out = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            out[column] = series.map(
                lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}"
            )
        elif pd.api.types.is_integer_dtype(series):
            out[column] = series.map(lambda v: "" if pd.isna(v) else f"{v:,}".replace(",", " "))
        else:
            out[column] = series.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    return out


def to_markdown(frame: pd.DataFrame, decimals: int = 4) -> str:
    """Render a dataframe as a GitHub-flavoured Markdown table."""
    text = format_frame(frame, decimals=decimals)
    headers = [str(c) for c in text.columns]
    rows = text.astype(str).to_numpy().tolist()

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells: list[str]) -> str:
        padded = [cells[i].ljust(widths[i]) for i in range(len(cells))]
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), separator, *(line(r) for r in rows)])


def write(frame: pd.DataFrame, path: Path, decimals: int = 4, title: str = "") -> Path:
    """Write a dataframe to a Markdown file, with an optional heading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = to_markdown(frame, decimals=decimals)
    content = f"# {title}\n\n{body}\n" if title else body + "\n"
    path.write_text(content, encoding="utf-8")
    return path
