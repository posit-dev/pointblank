#!/usr/bin/env python3
"""
Script to generate llms.txt and llms-full.txt files for the Pointblank documentation.

This script can be run standalone without importing the full pointblank package.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from pointblank
sys.path.insert(0, str(Path(__file__).parent.parent))

from pointblank._utils_llms_txt import (
    _get_api_and_examples_text,
    generate_llms_full_txt,
    generate_llms_txt,
)


def main():
    """Generate both llms.txt and llms-full.txt files."""
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / "docs"
    data_dir = base_dir / "pointblank" / "data"

    # Ensure directories exist
    docs_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # First, regenerate the api-docs.txt file (used by assistant() and as cache for llms-full.txt)
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

    # Generate llms.txt
    print("\nGenerating llms.txt...")
    try:
        llms_content = generate_llms_txt()
        llms_path = docs_dir / "llms.txt"
        with open(llms_path, "w") as f:
            f.write(llms_content)
        print(f"✓ Generated {llms_path}")
    except Exception as e:
        print(f"✗ Failed to generate llms.txt: {e}")
        import traceback

        traceback.print_exc()

    # Generate llms-full.txt
    print("\nGenerating llms-full.txt...")
    try:
        llms_full_path = docs_dir / "llms-full.txt"
        generate_llms_full_txt(str(llms_full_path))
        print(f"✓ Generated {llms_full_path}")
    except Exception as e:
        print(f"✗ Failed to generate llms-full.txt: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
