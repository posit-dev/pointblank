import os
import glob
import re

# Print the working directory
print("Current working directory:", os.getcwd())

# Get a list of all files in the working directory
files = os.listdir(".")
print("Files in working directory:", files)

site_files = os.listdir("_site")
print("Files in '_site' directory:", site_files)

# Process all HTML files in the `_site/reference/` directory (except `index.html`)
# and apply the specified transformations
html_files = [f for f in glob.glob("_site/reference/*.html") if not f.endswith("index.html")]

print(f"Found {len(html_files)} HTML files to process")

for html_file in html_files:
    print(f"Processing: {html_file}")

    with open(html_file, "r") as file:
        content = file.readlines()

    # Determine the classification of each h1 tag based on its content
    classification_info = {}
    for i, line in enumerate(content):
        # Look for both class="title" and styled h1 tags
        h1_match = re.search(r'<h1\s+class="title">(.*?)</h1>', line)
        if not h1_match:
            # Also check for h1 tags with style attribute (for level1 section titles)
            h1_match = re.search(r'<h1\s+style="[^"]*">(.*?)</h1>', line)

        if h1_match:
            original_h1_content = h1_match.group(1).strip()
            # Store classification based on original content
            if original_h1_content and original_h1_content[0].isupper():
                if "." in original_h1_content:
                    classification_info[i] = ("method", "steelblue", "#E3F2FF")
                else:
                    classification_info[i] = ("class", "darkgreen", "#E3FEE3")
            else:
                classification_info[i] = ("function", "darkorange", "#FFF1E0")

    # Remove the literal text `Validate.` from the h1 tag
    # TODO: Add line below stating the class name for the method
    content = [
        line.replace(
            '<h1 class="title">Validate.',
            '<h1 class="title">',
        )
        for line in content
    ]

    # If the inner content of the h1 tag either:
    # - has a literal `.` in it, or
    # - doesn't start with a capital letter,
    # then add `()` to the end of the content of the h1 tag
    for i, line in enumerate(content):
        # Use regex to find h1 tags (both class="title" and styled versions)
        h1_match = re.search(r'<h1\s+class="title">', line)
        if not h1_match:
            h1_match = re.search(r'<h1\s+style="[^"]*">', line)

        if h1_match:
            # Extract the content of the h1 tag
            start = h1_match.end()
            end = line.find("</h1>", start)
            h1_content = line[start:end].strip()

            # Check if the content meets the criteria
            if "." in h1_content or (h1_content and not h1_content[0].isupper()):
                # Modify the content
                h1_content += "()"

            # Replace the h1 tag with the modified content
            content[i] = line[:start] + h1_content + line[end:]

    # Add classification labels using stored info
    for i, line in enumerate(content):
        if i in classification_info:
            h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", line)
            if h1_match:
                h1_content = h1_match.group(1)
                label_type, label_color, background_color = classification_info[i]

                label_span = f'<span style="font-size: 0.75rem; border-style: solid; border-width: 1px; border-color: {label_color}; background-color: {background_color}; margin-left: 12px; vertical-align: 3.5px;"><code style="background-color: transparent; color: {label_color};">{label_type}</code></span>'

                new_h1_content = h1_content + label_span
                new_line = line.replace(h1_content, new_h1_content)
                content[i] = new_line

    # Wrap bare h1 tags (those with style attribute but no quarto-title wrapper) in proper structure
    for i, line in enumerate(content):
        # Look for h1 tags with style attribute that aren't already wrapped
        if "<h1 style=" in line and "SFMono-Regular" in line:
            # Check if this h1 is already wrapped in quarto-title div
            # Look at previous lines to see if there's a quarto-title div
            is_wrapped = False
            for j in range(max(0, i - 5), i):
                if 'class="quarto-title"' in content[j]:
                    is_wrapped = True
                    break

            # If not wrapped, wrap it
            if not is_wrapped:
                # Extract the h1 content
                h1_content = line.strip()

                # Replace the line with the wrapped version
                wrapped_h1 = f'<div class="quarto-title">\n{h1_content}\n</div>\n'
                content[i] = wrapped_h1

    # Add a style attribute to the h1 tag to use a monospace font for code-like appearance
    content = [
        line.replace(
            '<h1 class="title">',
            "<h1 class=\"title\" style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 1.25rem;\">",
        )
        for line in content
    ]

    # Some h1 tags may not have a class attribute, so we handle that case too
    content = [
        line.replace(
            "<h1>",
            "<h1 style=\"font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size: 1.25rem;\">",
        )
        for line in content
    ]

    # Move the first <p> tag (description) to immediately after the title header
    header_end_line = None
    first_p_line = None
    first_p_content = None
    found_sourcecode = False
    title_line = None
    sourcecode_line = None

    # First pass: find the header end, title, sourcecode, and the first <p> tag after sourceCode
    for i, line in enumerate(content):
        # Find where the header ends
        if "</header>" in line:
            header_end_line = i

        # Find the title line (either in header or in level1 section)
        if '<h1 class="title"' in line or ("<h1 style=" in line and "SFMono-Regular" in line):
            title_line = i

        # Look for the sourceCode div
        if '<div class="sourceCode" id="cb1">' in line:
            found_sourcecode = True
            sourcecode_line = i

        # Find the first <p> tag after we've seen the sourceCode div
        if found_sourcecode and first_p_line is None and line.strip().startswith("<p"):
            first_p_line = i
            first_p_content = line
            break

    # Determine where to insert the description paragraph
    # If title is after header, insert after title; otherwise insert after header
    if (
        header_end_line is not None
        and first_p_line is not None
        and title_line is not None
        and sourcecode_line is not None
    ):
        if title_line > header_end_line:
            # Title is in a separate section, insert after title
            insert_after_line = title_line
        else:
            # Title is in header, insert after header
            insert_after_line = header_end_line

        # Apply italic styling to the description
        if "style=" not in first_p_content:
            styled_p = first_p_content.replace(
                "<p>",
                '<p style="font-size: 1rem; font-style: italic; margin-top: -10px; line-height: 1;">',
            )
        else:
            styled_p = first_p_content

        # Remove the original <p> line
        content.pop(first_p_line)

        # Adjust sourcecode_line since we removed a line before it
        if first_p_line < sourcecode_line:
            sourcecode_line -= 1

        # Insert the styled <p> line after the determined position (accounting for the removed line)
        insert_position = (
            insert_after_line + 1 if first_p_line > insert_after_line else insert_after_line
        )
        content.insert(insert_position, "\n")  # Add spacing
        content.insert(insert_position + 1, styled_p)
        content.insert(insert_position + 2, "\n")  # Add spacing

        # Adjust sourcecode_line since we added lines before it
        sourcecode_line += 3

        # Add "USAGE" label before the sourceCode div
        usage_label = '<p style="font-size: 12px; color: rgb(170, 170, 170); margin-bottom: -14px;">USAGE</p>\n'
        content.insert(sourcecode_line, usage_label)

    # Style the first and second <dl> tags with different borders
    dl_count = 0
    for i, line in enumerate(content):
        if "<dl>" in line:
            dl_count += 1
            if dl_count == 1:
                # First <dl> tag - green border
                content[i] = line.replace(
                    "<dl>",
                    '<dl style="border-style: solid; border-width: 2px; border-color: #00AC1480; padding: 1rem; padding-bottom: 0.25rem;">',
                )
            elif dl_count == 2:
                # Second <dl> tag - indigo border
                content[i] = line.replace(
                    "<dl>",
                    '<dl style="border-style: solid; border-width: 2px; border-color: #0059AC80; padding: 1rem; padding-bottom: 0.25rem;">',
                )
                break  # Stop after finding the second one

    # Fix return value formatting in individual function pages, removing the `:` before the
    # return value and adjusting the style of the parameter annotation separator
    content_str = "".join(content)
    return_value_pattern = (
        r'<span class="parameter-name"></span> <span class="parameter-annotation-sep">:</span>'
    )
    return_value_replacement = r'<span class="parameter-name"></span> <span class="parameter-annotation-sep" style="margin-left: -8px;"></span>'
    content_str = re.sub(return_value_pattern, return_value_replacement, content_str)

    # Fix double asterisks in kwargs parameters
    content_str = content_str.replace("****kwargs**", "**<strong>kwargs</strong>")

    content = content_str.splitlines(keepends=True)

    # Turn all h3 tags into h4 tags
    content = [line.replace("<h3", "<h4").replace("</h3>", "</h4>") for line in content]

    # Turn all h2 tags into h3 tags
    content = [line.replace("<h2", "<h3").replace("</h2>", "</h3>") for line in content]

    # Add gradient animation to Examples headers and horizontal rules
    content_str = "".join(content)

    # Find and replace Examples headers with animated gradient styling
    examples_pattern = (
        r'(<h3[^>]*class="[^"]*doc-section-examples[^"]*"[^>]*>)(.*?Examples.*?)(</h3>)'
    )
    examples_replacement = r"""\1<span style="
        background: linear-gradient(-45deg, #D63031, #00B894, #0984E3, #6C5CE7, #FDCB6E, #A29BFE, #E84393, #2D3436);
        background-size: 400% 400%;
        animation: examplesGradient 8s ease-in-out infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        font-size: 1.2em;
    ">\2</span>\3
    <hr style="
        background: linear-gradient(-45deg, #D63031, #00B894, #0984E3, #6C5CE7, #FDCB6E, #A29BFE, #E84393, #2D3436);
        background-size: 400% 400%;
        animation: examplesGradient 16s ease-in-out infinite;
        height: 3px;
        border: none;
        margin: 10px 0;
        border-radius: 2px;
    ">"""

    content_str = re.sub(examples_pattern, examples_replacement, content_str, flags=re.DOTALL)

    content = content_str.splitlines(keepends=True)

    # Place a horizontal rule at the end of each reference page
    content_str = "".join(content)
    main_end_pattern = r"</main>"
    main_end_replacement = (
        "</main>\n"
        '<hr style="padding: 0; margin: 0;">\n'
        '<div style="text-align: center; padding: 0; margin-top: -60px; color: #B3B3B3;">⦾</div>'
    )
    content_str = re.sub(main_end_pattern, main_end_replacement, content_str)
    content = content_str.splitlines(keepends=True)

    with open(html_file, "w") as file:
        file.writelines(content)


# Modify the `index.html` file in the `_site/reference/` directory
index_file = "_site/reference/index.html"

if os.path.exists(index_file):
    print(f"Processing index file: {index_file}")

    with open(index_file, "r") as file:
        content = file.read()

    # Convert tables to dl/dt/dd format
    def convert_table_to_dl(match):
        table_content = match.group(1)

        # Extract all table rows
        row_pattern = r"<tr[^>]*>(.*?)</tr>"
        rows = re.findall(row_pattern, table_content, re.DOTALL)

        dl_items = []
        for row in rows:
            # Extract the two td elements
            td_pattern = r"<td[^>]*>(.*?)</td>"
            tds = re.findall(td_pattern, row, re.DOTALL)

            if len(tds) == 2:
                link_content = tds[0].strip()
                description = tds[1].strip()

                dt = f"<dt>{link_content}</dt>"
                dd = f'<dd style="text-indent: 20px; margin-top: -3px;">{description}</dd>'
                dl_items.append(f"{dt}\n{dd}")

        dl_content = "\n\n".join(dl_items)
        return f'<div class="caption-top table" style="border-top-style: dashed; border-bottom-style: dashed;">\n<dl style="margin-top: 10px;">\n\n{dl_content}\n\n</dl>\n</div>'

    # Replace all table structures with dl/dt/dd
    table_pattern = r'<table class="caption-top table">\s*<tbody>(.*?)</tbody>\s*</table>'
    content = re.sub(table_pattern, convert_table_to_dl, content, flags=re.DOTALL)

    # Add () to methods and functions in <a> tags within <dt> elements
    def add_parens_to_functions(match):
        full_tag = match.group(0)
        link_text = match.group(1)

        # Rules for adding ():
        # - Don't touch capitalized content (classes)
        # - Add () if text has a period (methods like Validate.col_vals_gt)
        # - Add () if text doesn't start with capital (functions like starts_with, load_dataset)
        if "." in link_text or (link_text and not link_text[0].isupper()):
            # Replace the link text with the same text + ()
            return full_tag.replace(f">{link_text}</a>", f">{link_text}()</a>")

        return full_tag

    # Find all <a> tags within <dt> elements and apply the function
    dt_link_pattern = r"<dt><a[^>]*>([^<]+)</a></dt>"
    content = re.sub(dt_link_pattern, add_parens_to_functions, content)

    # Remove redundant "API Reference" top-level nav item
    # Find the nav structure and flatten it by removing the top-level wrapper
    nav_pattern = r'(<nav[^>]*>.*?<h2[^>]*>.*?</h2>\s*<ul>\s*)<li><a[^>]*href="[^"]*#api-reference"[^>]*>API Reference</a>\s*<ul[^>]*>(.*?)</ul></li>\s*(</ul>\s*</nav>)'
    nav_replacement = r"\1\2\3"
    content = re.sub(nav_pattern, nav_replacement, content, flags=re.DOTALL)

    with open(index_file, "w") as file:
        file.write(content)

    print("Index file processing complete")
else:
    print(f"Index file not found: {index_file}")


# Update quarto-secondary-nav-title to display "User Guide" text
# This improves the mobile navigation by making it clear what the sidebar toggle reveals
all_html_files = glob.glob("_site/**/*.html", recursive=True)
print(f"Found {len(all_html_files)} HTML files to check for secondary nav title")

for html_file in all_html_files:
    with open(html_file, "r") as file:
        content = file.read()

    # Replace empty h1.quarto-secondary-nav-title with h5 containing "User Guide"
    original_pattern = r'<h1 class="quarto-secondary-nav-title"></h1>'
    replacement = '<h5 class="quarto-secondary-nav-title">User Guide</h5>'

    if original_pattern in content:
        print(f"Updating secondary nav title in: {html_file}")
        content = content.replace(original_pattern, replacement)

        with open(html_file, "w") as file:
            file.write(content)


print("Finished processing all files")
