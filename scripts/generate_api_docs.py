#!/usr/bin/env python3
"""
Script to regenerate the api-docs.txt file used by DraftValidation and assistant().

The llms.txt and llms-full.txt files are now generated natively by Great Docs during
the site build process.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from pointblank
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointblank._utils_llms_txt import _get_api_and_examples_text


def main():
    """Regenerate the api-docs.txt file."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "pointblank" / "data"

    # Ensure directory exists
    data_dir.mkdir(exist_ok=True)

    # Regenerate the api-docs.txt file (used by DraftValidation and assistant())
    print("Regenerating api-docs.txt...")
    try:
        api_docs_content = _get_api_and_examples_text()
        api_docs_path = data_dir / "api-docs.txt"
        with open(api_docs_path, "w") as f:
            f.write(api_docs_content)
        print(f"✓ Generated {api_docs_path}")
    except Exception as e:
        print(f"✗ Failed to generate api-docs.txt: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
