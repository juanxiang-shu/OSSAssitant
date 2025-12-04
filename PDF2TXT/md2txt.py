import os
import re
from pathlib import Path


def clean_scientific_markdown(content: str) -> str:
    """
    Clean scientific literature-style Markdown content:
    - keep math/LaTeX formulas
    - remove most Markdown/HTML formatting while preserving text
    """

    # Remove XML/HTML tags (e.g., <document>, <p>, etc.)
    content = re.sub(r'<[^>]+>', '', content)

    # Remove image links but keep the alt text / caption
    # Match ![alt text](url) and keep only the alt text
    content = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', content)

    # Remove normal link markup but keep link text
    # Match [link text](url) and keep only the link text
    content = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', content)

    # Remove heading '#' markers but keep heading text
    content = re.sub(r'^#{1,6}\s*', '', content, flags=re.MULTILINE)

    # Remove bold/italic markers while keeping the text
    content = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', content)
    content = re.sub(r'_{1,3}([^_]+)_{1,3}', r'\1', content)

    # Remove strikethrough markers while keeping the text
    content = re.sub(r'~~([^~]+)~~', r'\1', content)

    # Remove fenced code block markers but keep the code content
    content = re.sub(r'```[a-zA-Z]*\n?', '', content)
    content = re.sub(r'`([^`]+)`', r'\1', content)

    # Remove quote markers (">") at line starts
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)

    # Remove list markers but keep the list text
    content = re.sub(r'^[\s]*[-*+]\s+', '', content, flags=re.MULTILINE)
    content = re.sub(r'^\s*\d+\.\s+', '', content, flags=re.MULTILINE)

    # Remove simple table separators
    content = re.sub(r'\|', ' ', content)
    content = re.sub(r'^[-:\s]+$', '', content, flags=re.MULTILINE)

    # Remove horizontal rules
    content = re.sub(r'^[-*_]{3,}$', '', content, flags=re.MULTILINE)

    # Normalize excessive blank lines: keep at most two consecutive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Strip leading/trailing whitespace on each line
    lines = content.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    content = '\n'.join(cleaned_lines)

    # Remove leading and trailing blank lines
    content = content.strip()

    return content


def process_md_files(input_dir: str, output_dir: str) -> None:
    """
    Process all Markdown files in the given directory and write cleaned TXT files.

    Args:
        input_dir (str): Input directory path containing .md files.
        output_dir (str): Output directory path for cleaned .txt files.
    """

    # Create Path objects for input and output
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory {input_dir} does not exist!")
        return

    # Create output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all .md files
    md_files = list(input_path.glob('*.md'))

    if not md_files:
        print(f"No .md files found in directory {input_dir}.")
        return

    print(f"Found {len(md_files)} Markdown files. Start processing...")

    # Process each file
    success_count = 0
    for md_file in md_files:
        try:
            # Read original file
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Clean content
            cleaned_content = clean_scientific_markdown(content)

            # Generate output filename (.md -> .txt)
            output_filename = md_file.stem + '.txt'
            output_file = output_path / output_filename

            # Save cleaned content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)

            print(f"✓ Processed: {md_file.name} -> {output_filename}")
            success_count += 1

        except Exception as e:
            print(f"✗ Error processing file {md_file.name}: {str(e)}")

    print(f"\nDone! Successfully processed {success_count}/{len(md_files)} files.")
    print(f"Cleaned files are saved in: {output_dir}")


def main() -> None:
    """
    Entry point for the Markdown cleaning script.
    Set your own input and output directories here before running.
    """
    # Set input and output directories (edit these paths before use)
    input_directory = r"path/to/your/input_md_folder"
    output_directory = r"path/to/your/output_txt_folder"

    print("Scientific Markdown Cleaning Tool")
    print("=" * 50)
    print(f"Input directory:  {input_directory}")
    print(f"Output directory: {output_directory}")
    print("=" * 50)

    # Process files
    process_md_files(input_directory, output_directory)


if __name__ == "__main__":
    main()