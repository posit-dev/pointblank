#!/usr/bin/env python3
"""
Create a version of user-guide-pdf.qmd that includes content without YAML front matter.
"""

import re
from pathlib import Path


def strip_yaml_frontmatter(content: str) -> str:
    """Remove YAML front matter from content."""
    # Match YAML front matter (--- at start, content, --- or ... at end)
    pattern = r"^---\s*\n.*?\n(?:---|\.\.\.)[ \t]*\n"
    return re.sub(pattern, "", content, count=1, flags=re.DOTALL | re.MULTILINE)


def process_include(include_line: str, base_path: Path) -> str:
    """Process an include directive and return content without YAML."""
    # Extract file path from {{< include path >}}
    match = re.search(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}", include_line)
    if not match:
        return include_line

    file_path = base_path / match.group(1)

    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return include_line

    # Read and strip YAML
    content = file_path.read_text()
    content = strip_yaml_frontmatter(content)

    return content


def create_pdf_version():
    """Create user-guide-pdf-clean.qmd with YAML stripped from includes."""
    docs_dir = Path(__file__).parent.parent / "docs"
    source_file = docs_dir / "user-guide-pdf.qmd"
    output_file = docs_dir / "user-guide-pdf-clean.qmd"

    if not source_file.exists():
        print(f"Error: {source_file} not found")
        return

    content = source_file.read_text()
    lines = content.split("\n")

    output_lines = []
    in_frontmatter = False
    frontmatter_done = False

    for i, line in enumerate(lines):
        # Keep the main document's YAML front matter
        if i == 0 and line.strip() == "---":
            in_frontmatter = True
            output_lines.append(line)
            continue

        if in_frontmatter:
            output_lines.append(line)
            if line.strip() in ["---", "..."]:
                in_frontmatter = False
                frontmatter_done = True
            continue

        # Process include directives
        if "{{< include" in line:
            processed = process_include(line, docs_dir)
            output_lines.append(processed)
        else:
            output_lines.append(line)

    output_file.write_text("\n".join(output_lines))
    print(f"Created {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    create_pdf_version()
