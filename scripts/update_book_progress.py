#!/usr/bin/env python3
"""
Update README.md with chapter progress from the Pointblank book repo.

Usage:
    python scripts/update_book_progress.py

The script clones (or pulls) the book repo into a temporary directory, reads
_quarto.yml for the chapter structure, measures content length per chapter,
and writes a progress table into README.md between sentinel comments.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import yaml

BOOK_REPO = "https://github.com/rich-iannone/pointblank-the-complete-guide.git"
TARGET_CHARS = 15_000  # expected character count for a "complete" chapter
BAR_WIDTH = 15  # number of block characters in a progress bar

FILLED = "█"
EMPTY = "░"

BEGIN_SENTINEL = "<!-- BOOK_PROGRESS_BEGIN -->"
END_SENTINEL = "<!-- BOOK_PROGRESS_END -->"

# Words that should stay uppercase in chapter titles
UPPERCASE_WORDS = {"ai", "cdisc", "mcp", "yaml", "rwe", "ml", "cli"}
SPECIAL_CASE_WORDS = {"iot": "IoT"}

# Filename stem -> display title overrides (after stripping number prefix and playbook- prefix)
TITLE_OVERRIDES = {
    "thresholds-actions": "Thresholds & Actions",
    "reports-extracts": "Reports & Extracts",
    "notifications-observability": "Notifications & Observability",
    "clinical-cdisc": "Clinical & CDISC",
    "cdisc-conformance": "CDISC Conformance",
    "ai-authoring": "AI Authoring",
    "data-engineering": "Data Engineering",
    "ml-monitoring": "ML Monitoring",
    "iot-sensors": "IoT Sensors",
    "public-sector": "Public Sector",
    "rwe": "Real-World Evidence",
}


def clone_or_pull(dest: Path) -> None:
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "-q"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--depth=1", BOOK_REPO, str(dest)], check=True)


def content_chars(path: Path) -> int:
    """Return character count of a .qmd file, excluding hidden code blocks."""
    text = path.read_text(encoding="utf-8")

    # Strip hidden setup blocks: ```{python}\n#| include: false\n...\n```
    text = re.sub(
        r"```\{python\}\s*\n#\|\s*include:\s*false\n.*?```",
        "",
        text,
        flags=re.DOTALL,
    )

    # Strip the chapter heading line
    text = re.sub(r"^#[^\n]*\n", "", text)

    return len(text.strip())


def nice_title(filename: str) -> str:
    """Derive a readable chapter title from a filename like '03-inspecting-data.qmd'."""
    stem = Path(filename).stem
    # Remove leading number prefix
    stem = re.sub(r"^\d+-", "", stem)
    # Remove 'playbook-' prefix for cleaner display
    key = re.sub(r"^playbook-", "", stem)

    if key in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[key]

    words = key.split("-")
    titled = []
    for w in words:
        lower = w.lower()
        if lower in SPECIAL_CASE_WORDS:
            titled.append(SPECIAL_CASE_WORDS[lower])
        elif lower in UPPERCASE_WORDS:
            titled.append(w.upper())
        else:
            titled.append(w.capitalize())
    return " ".join(titled)


def progress_bar(chars: int) -> str:
    pct = min(chars / TARGET_CHARS, 1.0)
    filled = round(pct * BAR_WIDTH)
    bar = FILLED * filled + EMPTY * (BAR_WIDTH - filled)
    pct_display = min(int(pct * 100), 100)
    return f"`{bar}` {pct_display}%"


def build_progress_md(book_dir: Path) -> str:
    quarto_cfg = yaml.safe_load((book_dir / "_quarto.yml").read_text(encoding="utf-8"))

    chapters_cfg = quarto_cfg["book"]["chapters"]

    lines: list[str] = []

    # Collect all files for overall stats
    all_files: list[str] = []

    for entry in chapters_cfg:
        if isinstance(entry, str):
            continue

        part_name = entry["part"]
        chapter_files = entry["chapters"]
        all_files.extend(chapter_files)

        lines.append(f"**{part_name}**<br>")

        for fname in chapter_files:
            fpath = book_dir / fname
            chars = content_chars(fpath) if fpath.exists() else 0
            title = nice_title(fname)
            bar = progress_bar(chars)
            lines.append(f"{title}: {bar}<br>")

        lines.append("")

    total_chars = sum(content_chars(book_dir / f) for f in all_files if (book_dir / f).exists())
    total_possible = len(all_files) * TARGET_CHARS
    overall_pct = min(int(total_chars / total_possible * 100), 100)

    written = len(
        [f for f in all_files if (book_dir / f).exists() and content_chars(book_dir / f) >= 1000]
    )

    header = (
        f"Overall: **{overall_pct}%** complete "
        f"&mdash; {written} of {len(all_files)} chapters have content"
    )

    return header + "\n\n" + "\n".join(lines)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    readme = repo_root / "README.md"

    with tempfile.TemporaryDirectory() as tmp:
        book_dir = Path(tmp) / "book"
        clone_or_pull(book_dir)
        progress_md = build_progress_md(book_dir)

    content = readme.read_text(encoding="utf-8")

    if BEGIN_SENTINEL in content and END_SENTINEL in content:
        replacement = f"{BEGIN_SENTINEL}\n{progress_md}\n{END_SENTINEL}"
        content = re.sub(
            rf"{re.escape(BEGIN_SENTINEL)}.*?{re.escape(END_SENTINEL)}",
            replacement,
            content,
            flags=re.DOTALL,
        )
    else:
        raise RuntimeError(
            f"Could not find sentinel comments in {readme}. "
            f"Ensure {BEGIN_SENTINEL} and {END_SENTINEL} are present."
        )

    readme.write_text(content, encoding="utf-8")
    print(f"Updated {readme} with book progress ({progress_md.splitlines()[0]})")


if __name__ == "__main__":
    main()
