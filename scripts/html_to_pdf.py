#!/usr/bin/env python3
"""
Convert HTML file to PDF using Playwright with custom print CSS.
This preserves all HTML content including validation reports with proper text selection.
"""

import subprocess
import sys
from pathlib import Path


def html_to_pdf_chrome(html_path: str, pdf_path: str):
    """Convert HTML to PDF using Chrome/Chromium headless."""
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path).resolve()

    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)

    print(f"Converting {html_path} to {pdf_path}...")

    # Try to find Chrome executable
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ]

    chrome_cmd = None
    for path in chrome_paths:
        if Path(path).exists():
            chrome_cmd = path
            break

    if not chrome_cmd:
        print("Error: Chrome/Chromium not found.")
        print("Please install Google Chrome or Chromium.")
        sys.exit(1)

    # Run Chrome headless to print to PDF
    result = subprocess.run(
        [
            chrome_cmd,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{html_path}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error converting to PDF: {result.stderr}")
        sys.exit(1)

    if pdf_path.exists():
        print(f"PDF generated successfully: {pdf_path}")
        print(f"Size: {pdf_path.stat().st_size / (1024 * 1024):.1f} MB")
    else:
        print("Error: PDF was not created")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: html_to_pdf.py <input.html> <output.pdf>")
        sys.exit(1)

    html_to_pdf_chrome(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: html_to_pdf.py <input.html> <output.pdf>")
        sys.exit(1)

    html_to_pdf_chrome(sys.argv[1], sys.argv[2])
