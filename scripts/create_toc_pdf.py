#!/usr/bin/env python3
"""
Extract page numbers from PDF and create a Table of Contents page.
This script analyzes the generated PDF to find actual page numbers for each section.
"""

import subprocess
import sys
from pathlib import Path


def find_section_pages_from_pdf(pdf_path: Path) -> dict[str, int]:
    """
    Extract actual page numbers by analyzing the PDF content using PyPDF2.
    Looks for the section headings in the PDF text.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("Installing PyPDF2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2"], check=True)
        from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    section_pages = {}
    # Search patterns - these match the actual H1 sections in the user guide
    sections_to_find = [
        ("1 Validation Plan", "validation-plan"),
        ("2 Advanced Validation", "advanced-validation"),
        ("3 YAML", "yaml"),
        ("4 Post Interrogation", "post-interrogation"),
        ("5 Data Inspection", "data-inspection"),
        ("6 The Pointblank CLI", "pointblank-cli"),
    ]

    print(f"Scanning {total_pages} pages for section headings...")

    for page_num in range(total_pages):
        try:
            page = reader.pages[page_num]
            page_text = page.extract_text()

            # Check if any section heading appears on this page
            for section_heading, section_id in sections_to_find:
                # Look for the pattern at the start of a line (after whitespace)
                if section_heading in page_text and section_id not in section_pages:
                    # Page numbers are 1-indexed for display
                    section_pages[section_id] = page_num + 1
                    print(f"Found '{section_heading}' on page {page_num + 1}")
        except Exception as e:
            print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
            continue

    return section_pages


def create_toc_html(section_pages: dict[str, int], output_path: Path) -> None:
    """Create a standalone TOC HTML page."""

    sections = [
        ("validation-plan", "1", "Validation Plan"),
        ("advanced-validation", "2", "Advanced Validation"),
        ("yaml", "3", "YAML"),
        ("post-interrogation", "4", "Post Interrogation"),
        ("data-inspection", "5", "Data Inspection"),
        ("pointblank-cli", "6", "The Pointblank CLI"),
    ]

    toc_entries = []
    for section_id, num, title in sections:
        # Get the page number from the original PDF
        original_page = section_pages.get(section_id, "...")

        # Adjust for display: original page N becomes display page N-1
        # (subtract 1 for title page; TOC is inserted separately and blank page exists)
        if original_page != "...":
            display_page = original_page - 1
            # Store the original page for link destination (will be offset by 1 after TOC insertion)
            link_page = original_page
        else:
            display_page = "..."
            link_page = None

        # Create clickable link if we have a valid page number
        if link_page:
            toc_entries.append(f"""
        <a href="#page={link_page}" class="toc-link">
            <div class="toc-entry">
                <div class="toc-text">{num}. {title}</div>
                <div class="toc-dots"></div>
                <div class="toc-page">{display_page}</div>
            </div>
        </a>
        """)
        else:
            toc_entries.append(f"""
        <div class="toc-entry">
            <div class="toc-text">{num}. {title}</div>
            <div class="toc-dots"></div>
            <div class="toc-page">{display_page}</div>
        </div>
        """)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {{
            size: letter landscape;
            margin: 0.5in 0.5in 0.75in 0.5in;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 1in 3in;
            background: white;
        }}

        h1 {{
            font-size: 24pt;
            text-align: center;
            margin-bottom: 1.5em;
            margin-top: 0.5in;
            color: #333;
        }}

        .toc-link {{
            text-decoration: none;
            color: inherit;
            display: block;
        }}

        .toc-link:hover {{
            background-color: #f5f5f5;
        }}

        .toc-entry {{
            display: flex;
            align-items: baseline;
            margin: 1em 0;
            font-size: 14pt;
            line-height: 1.8;
        }}

        .toc-text {{
            flex-shrink: 0;
            padding-right: 0.5em;
            color: #333;
        }}

        .toc-dots {{
            flex-grow: 1;
            border-bottom: 1px dotted #999;
            margin: 0 0.5em;
            height: 0.6em;
        }}

        .toc-page {{
            flex-shrink: 0;
            padding-left: 0.5em;
            font-weight: 500;
            color: #333;
        }}
    </style>
</head>
<body>
    <h1>Table of Contents</h1>
    {"".join(toc_entries)}
</body>
</html>"""

    output_path.write_text(html)
    print(f"Created TOC HTML at {output_path}")


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Convert HTML to PDF using Chrome headless."""
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ]

    chrome = None
    for path in chrome_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, check=True)
            chrome = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not chrome:
        print("Error: Chrome/Chromium not found")
        sys.exit(1)

    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{html_path.absolute()}",
        ],
        check=True,
    )

    print(f"Created TOC PDF at {pdf_path}")


def add_links_to_toc_pdf(toc_pdf_path: Path, section_pages: dict[str, int]) -> None:
    """Add clickable link annotations to the TOC PDF that point to section pages."""
    try:
        from PyPDF2 import PdfReader, PdfWriter
        from PyPDF2.generic import (
            ArrayObject,
            DictionaryObject,
            FloatObject,
            NameObject,
            NumberObject,
        )
    except ImportError:
        print("Installing PyPDF2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2"], check=True)
        from PyPDF2 import PdfReader, PdfWriter
        from PyPDF2.generic import (
            ArrayObject,
            DictionaryObject,
            FloatObject,
            NameObject,
            NumberObject,
        )

    reader = PdfReader(str(toc_pdf_path))
    writer = PdfWriter()

    # TOC has only one page
    page = reader.pages[0]
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)

    # Define sections with their target pages and vertical positions
    # PDF coordinates are from BOTTOM-left corner
    # Landscape letter: width=792, height=612 points
    # Each entry is roughly 25 points tall (14pt * 1.8 line-height)
    # Listed in visual order (top to bottom) with Y coordinates from bottom
    sections = [
        ("validation-plan", 325),  # Top entry in TOC (highest Y from bottom)
        ("advanced-validation", 300),
        ("yaml", 275),
        ("post-interrogation", 250),
        ("data-inspection", 225),
        ("pointblank-cli", 200),  # Bottom entry in TOC (lowest Y from bottom)
    ]

    # Add link annotations for each TOC entry
    annotations = []
    for section_id, y_from_bottom in sections:
        if section_id in section_pages:
            # Target page in the final merged PDF (accounting for title + TOC pages)
            target_page = section_pages[section_id]  # This is the page index in final PDF

            # Create link annotation
            # The rectangle defines the clickable area (left, bottom, right, top)
            # Make the entire TOC line clickable (height of ~25 points)
            link_rect = ArrayObject(
                [
                    FloatObject(180),  # Left - start after left margin/padding
                    FloatObject(y_from_bottom),  # Bottom
                    FloatObject(page_width - 180),  # Right - end before right margin/padding
                    FloatObject(y_from_bottom + 25),  # Top (25 points tall)
                ]
            )

            # Create the link annotation dictionary
            link_dict = DictionaryObject()
            link_dict.update(
                {
                    NameObject("/Type"): NameObject("/Annot"),
                    NameObject("/Subtype"): NameObject("/Link"),
                    NameObject("/Rect"): link_rect,
                    NameObject("/Border"): ArrayObject(
                        [NumberObject(0), NumberObject(0), NumberObject(0)]
                    ),
                    NameObject("/A"): DictionaryObject(
                        {
                            NameObject("/S"): NameObject("/GoTo"),
                            NameObject("/D"): ArrayObject(
                                [
                                    NumberObject(target_page),  # Page index
                                    NameObject("/XYZ"),  # Keep current zoom
                                    NumberObject(0),  # X position (left)
                                    NumberObject(page_height),  # Y position (top)
                                    NumberObject(0),  # Zoom (0 = keep current)
                                ]
                            ),
                        }
                    ),
                }
            )

            annotations.append(link_dict)

    # Add all annotations to the page
    if "/Annots" in page:
        # Extend existing annotations
        for annot in annotations:
            page["/Annots"].append(writer._add_object(annot))
    else:
        # Create new annotations array
        page[NameObject("/Annots")] = ArrayObject(
            [writer._add_object(annot) for annot in annotations]
        )

    writer.add_page(page)

    # Write back to file
    with open(toc_pdf_path, "wb") as f:
        writer.write(f)

    print(f"Added {len(annotations)} clickable links to TOC")


def add_page_numbers_to_pdf(pdf_path: Path) -> None:
    """Add page numbers to the bottom center of each page (except title page)."""
    try:
        import io

        from PyPDF2 import PdfReader, PdfWriter
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.pdfgen import canvas
    except ImportError:
        print("Installing required packages...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2", "reportlab"], check=True)
        import io

        from PyPDF2 import PdfReader, PdfWriter
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.pdfgen import canvas

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    page_width, page_height = landscape(letter)

    for page_num, page in enumerate(reader.pages):
        # Skip title page (page 0) and TOC (page 1)
        if page_num < 2:
            writer.add_page(page)
            continue

        # Create a new PDF with just the page number
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=landscape(letter))

        # Add page number at bottom center
        # Page numbers start from 1 for the first content page (after title + TOC)
        display_page_num = page_num - 1  # Subtract 1 because TOC is page 2
        can.setFont("Helvetica", 10)
        can.setFillColorRGB(0.4, 0.4, 0.4)  # Gray color
        text_width = can.stringWidth(str(display_page_num), "Helvetica", 10)
        can.drawString((page_width - text_width) / 2, 0.25 * 72, str(display_page_num))

        can.save()

        # Move to the beginning of the BytesIO buffer
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # Merge the overlay with the page
        page.merge_page(overlay_pdf.pages[0])
        writer.add_page(page)

    # Write to temporary file then replace original
    temp_path = pdf_path.parent / f"{pdf_path.stem}_temp.pdf"
    with open(temp_path, "wb") as f:
        writer.write(f)

    import shutil

    shutil.move(str(temp_path), str(pdf_path))
    print(f"Added page numbers to {len(reader.pages) - 2} pages")


def merge_pdfs(
    title_pdf: Path, toc_pdf: Path, main_pdf: Path, output_pdf: Path, section_pages: dict[str, int]
) -> None:
    """Merge title page, TOC, and main content PDFs using PyPDF2 and add bookmarks."""
    try:
        from PyPDF2 import PdfReader, PdfWriter
    except ImportError:
        print("Installing PyPDF2...")
        subprocess.run([sys.executable, "-m", "pip", "install", "PyPDF2"], check=True)
        from PyPDF2 import PdfReader, PdfWriter

    writer = PdfWriter()
    main_reader = PdfReader(str(main_pdf))
    toc_reader = PdfReader(str(toc_pdf))

    # Add title page (first page only)
    writer.add_page(main_reader.pages[0])

    # Add TOC page
    toc_page_idx = 1
    for page in toc_reader.pages:
        writer.add_page(page)

    # Add rest of main content (skip title page)
    # Pages are now offset by 1 (because we inserted TOC)
    for i in range(1, len(main_reader.pages)):
        writer.add_page(main_reader.pages[i])

    # Add bookmarks for each section
    sections = [
        ("validation-plan", "Validation Plan"),
        ("advanced-validation", "Advanced Validation"),
        ("yaml", "YAML"),
        ("post-interrogation", "Post Interrogation"),
        ("data-inspection", "Data Inspection"),
        ("pointblank-cli", "The Pointblank CLI"),
    ]

    for section_id, title in sections:
        if section_id in section_pages:
            # Page indices need to account for title + TOC pages
            # Original page N is now at index N+1 (because TOC was inserted)
            page_idx = section_pages[section_id]  # This is the page in the final merged PDF
            writer.add_outline_item(title, page_idx, parent=None)

    with open(output_pdf, "wb") as f:
        writer.write(f)

    print(f"Merged PDF created at {output_pdf}")
    print(f"Added {len(section_pages)} bookmarks")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_toc_pdf.py <main-pdf-path>")
        sys.exit(1)

    main_pdf = Path(sys.argv[1])

    if not main_pdf.exists():
        print(f"Error: {main_pdf} not found")
        sys.exit(1)

    print(f"Analyzing {main_pdf} for section page numbers...")
    section_pages = find_section_pages_from_pdf(main_pdf)

    if not section_pages:
        print("Warning: No sections found. Using placeholder page numbers.")
        section_pages = {
            "validation-plan": "...",
            "validation-methods": "...",
            "table-transformations": "...",
            "action-levels": "...",
            "interrogation": "...",
            "configuring-pointblank": "...",
        }

    # Create TOC HTML
    toc_html = main_pdf.parent / "toc.html"
    create_toc_html(section_pages, toc_html)

    # Convert TOC to PDF
    toc_pdf = main_pdf.parent / "toc.pdf"
    html_to_pdf(toc_html, toc_pdf)

    # Add clickable links to TOC PDF (before merging)
    print("Adding clickable links to TOC...")
    add_links_to_toc_pdf(toc_pdf, section_pages)

    # Create final merged PDF with bookmarks
    output_pdf = main_pdf.parent / "user-guide-with-toc.pdf"
    merge_pdfs(main_pdf, toc_pdf, main_pdf, output_pdf, section_pages)

    # Replace original
    import shutil

    shutil.move(str(output_pdf), str(main_pdf))

    # Add page numbers
    print("Adding page numbers to PDF...")
    add_page_numbers_to_pdf(main_pdf)

    # Clean up
    toc_html.unlink(missing_ok=True)
    toc_pdf.unlink(missing_ok=True)

    print(f"\nFinal PDF with TOC and page numbers: {main_pdf}")
    print(f"Size: {main_pdf.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
