"""
Parse LaTeX exam documents into JSONL format.

Usage:
    python parse_latex.py

Files:
    input_update.tex      - LaTeX source (required, paste your Overleaf content here)
    metadata_update.jsonl - Original questions for metadata (optional, same order as LaTeX)
    output_update.jsonl   - Parsed output (created/overwritten)

If metadata_update.jsonl is present and non-empty, the script reads id, difficulty,
and topics from it to populate the output. Otherwise, these fields are null/empty.
"""

import re
import json
import sys
from datetime import date
from pathlib import Path


def load_metadata(metadata_path: Path) -> list[dict]:
    """
    Load metadata from JSONL file.
    
    Returns empty list if file doesn't exist, is empty, or has errors.
    """
    if not metadata_path.exists():
        return []
    
    # Try utf-8-sig first (handles BOM), fall back to utf-8
    try:
        content = metadata_path.read_text(encoding='utf-8-sig').strip()
    except UnicodeDecodeError:
        content = metadata_path.read_text(encoding='utf-8').strip()
    
    if not content:
        return []
    
    metadata = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            metadata.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  [!] Warning: Invalid JSON on line {i} of metadata file: {e}")
    
    return metadata


def extract_questions(latex_content: str) -> list[dict]:
    r"""
    Extract questions from LaTeX content.
    
    Expects format:
        % ===== QUESTION N =====
        % Answer: X (Letter: Y)
        \item
        [stem content]
        \begin{tabular}...
        [options]
        \end{tabular}
        % ===== END QUESTION N =====
    """
    # Pattern to match question blocks
    question_pattern = re.compile(
        r'% ===== QUESTION (\d+) =====\s*\n'
        r'% Answer: (\d+) \(Letter: ([A-Z])\)\s*\n'
        r'\\item\s*\n?'
        r'(.*?)'
        r'% ===== END QUESTION \1 =====',
        re.DOTALL
    )
    
    questions = []
    
    for match in question_pattern.finditer(latex_content):
        q_num = int(match.group(1))
        answer_key = int(match.group(2))
        answer_letter = match.group(3)
        body = match.group(4).strip()
        
        # Parse the question body into stem and choices
        stem, choices = parse_question_body(body, q_num)
        
        # Verify answer key matches letter
        expected_index = ord(answer_letter) - ord('A')
        if answer_key != expected_index:
            print(f"  [!] Warning Q{q_num}: Answer key {answer_key} doesn't match letter {answer_letter} (index {expected_index})")
        
        questions.append({
            "question_number": q_num,
            "answer_key": answer_key,
            "prompt": stem,
            "choices": choices
        })
    
    return questions


def parse_question_body(body: str, q_num: int = 0) -> tuple[str, list[str]]:
    r"""
    Parse question body into stem and choices.
    
    Handles TWO tabular formats:
    
    1. Simple 2-column format (standard choices):
        \begin{tabular}{@{}ll@{}}
        \textbf{A} & option content \\[9pt]
        \textbf{B} & another option \\[9pt]
        ...
        \end{tabular}
    
    2. Grid format (multi-column answer tables):
        \begin{center}
        \begin{tabular}{|c|c|c|c|}
        \hline
         & \textbf{Header 1} & \textbf{Header 2} & ... \\
        \hline
        \textbf{A} & val1 & val2 & val3 \\
        \textbf{B} & val1 & val2 & val3 \\
        ...
        \end{tabular}
        \end{center}
    """
    # Try simple tabular format first (2-column @{}ll@{})
    simple_pattern = re.compile(
        r'\\begin\{tabular\}\{@\{\}ll@\{\}\}\s*(.*?)\s*\\end\{tabular\}',
        re.DOTALL
    )
    simple_match = simple_pattern.search(body)
    
    if simple_match:
        # Everything before the tabular is the stem
        stem = body[:simple_match.start()].strip()
        tabular_content = simple_match.group(1)
        
        # Clean up stem - remove trailing whitespace and vspace commands
        stem = re.sub(r'\\vspace\{[^}]*\}\s*$', '', stem).strip()
        stem = re.sub(r'\\\\\[\d+pt\]\s*$', '', stem).strip()
        
        # Parse options using the simple tabular parser
        choices = parse_tabular_choices(tabular_content, q_num)
        
        # Clean up stem (preserve array structure for I/II/III blocks)
        stem = clean_latex_stem(stem)
        
        return stem, choices
    
    # Try grid tabular format (|c|c|...|c| with borders and headers)
    # This format is wrapped in \begin{center}...\end{center}
    grid_pattern = re.compile(
        r'\\begin\{center\}\s*'
        r'(?:\\renewcommand\{\\arraystretch\}\{[^}]+\}\s*)?'  # optional arraystretch
        r'\\begin\{tabular\}\{\|c(?:\|c)+\|\}\s*'  # {|c|c|...|c|}
        r'(.*?)'
        r'\\end\{tabular\}\s*'
        r'\\end\{center\}',
        re.DOTALL
    )
    grid_match = grid_pattern.search(body)
    
    if grid_match:
        # Everything before \begin{center} is the stem
        stem = body[:grid_match.start()].strip()
        tabular_content = grid_match.group(1)
        
        # Clean up stem
        stem = re.sub(r'\\vspace\{[^}]*\}\s*$', '', stem).strip()
        stem = re.sub(r'\\\\\[\d+pt\]\s*$', '', stem).strip()
        
        # Parse options using the grid tabular parser
        choices = parse_grid_tabular_choices(tabular_content, q_num)
        
        # Clean up stem
        stem = clean_latex_stem(stem)
        
        return stem, choices
    
    print(f"  [!] Warning Q{q_num}: Could not find tabular block (tried simple and grid formats)")
    return body.strip(), []


def parse_tabular_choices(tabular_content: str, q_num: int = 0) -> list[str]:
    r"""
    Parse choices from tabular content, handling multi-line options.
    
    Multi-line options in LaTeX look like:
        \textbf{A} & first part 
        $continued$ \\ & second part \\[8pt]
    
    The key insight: `\\ &` marks a continuation cell in the same row.
    We need to:
    1. Normalize all whitespace (including newlines) into single spaces
    2. Split on `\textbf{X} &` to get each option
    3. For each option, merge text before and after `\\ &` (continuation)
    4. Clean and return
    """
    # Step 1: Normalize whitespace - join all lines into one string
    # This handles physical line breaks in the source file
    content = re.sub(r'\s+', ' ', tabular_content).strip()
    
    # Step 2: Split by option markers \textbf{A} &, \textbf{B} &, etc.
    # We use a pattern that captures the letter for ordering
    option_pattern = re.compile(r'\\textbf\{([A-Z])\}\s*&\s*')
    
    # Find all option start positions
    option_starts = [(m.start(), m.end(), m.group(1)) for m in option_pattern.finditer(content)]
    
    if not option_starts:
        print(f"  [!] Warning Q{q_num}: No options found in tabular")
        return []
    
    choices = []
    
    for i, (start, content_start, letter) in enumerate(option_starts):
        # Get the content up to the next option (or end)
        if i + 1 < len(option_starts):
            option_content = content[content_start:option_starts[i + 1][0]]
        else:
            option_content = content[content_start:]
        
        # Step 3: Handle continuation cells - merge text around \\ &
        # Pattern: "first part \\ & second part" -> "first part second part"
        # The \\ & is a tabular row break within the same logical choice
        option_content = re.sub(r'\s*\\\\\s*&\s*', ' ', option_content)
        
        # Remove trailing \\ or \\[Xpt] (end of this choice's row)
        option_content = re.sub(r'\s*\\\\\s*(\[\d+pt\])?\s*$', '', option_content)
        
        # Clean and add
        cleaned = clean_latex_choice(option_content)
        choices.append(cleaned)
    
    # Validation: check for duplicate choices (indicates truncation bug)
    normalized_choices = [normalize_for_comparison(c) for c in choices]
    duplicates = find_duplicates(normalized_choices)
    if duplicates:
        print(f"  [!] Warning Q{q_num}: Duplicate choices detected (possible truncation): {duplicates}")
    
    return choices


def parse_grid_tabular_choices(tabular_content: str, q_num: int = 0) -> list[str]:
    r"""
    Parse choices from grid tabular content (multi-column answer tables).
    
    Grid format in LaTeX:
        \hline
         & \textbf{Header 1} & \textbf{Header 2} & \textbf{Header 3} \\
        \hline
        \textbf{A} & val1 & val2 & val3 \\
        \textbf{B} & val1 & val2 & val3 \\
        ...
        \hline
    
    Output: Each row becomes a comma-separated choice string:
        "val1, val2, val3"
    """
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', tabular_content).strip()
    
    # Remove \hline commands
    content = re.sub(r'\\hline\s*', '', content)
    
    # Split into rows by \\ (row separator)
    # Each row ends with \\ possibly followed by spacing like [2pt]
    rows = re.split(r'\s*\\\\\s*(?:\[\d+pt\])?\s*', content)
    rows = [r.strip() for r in rows if r.strip()]
    
    if not rows:
        print(f"  [!] Warning Q{q_num}: No rows found in grid tabular")
        return []
    
    # First row is the header (skip it - starts with & not \textbf{X})
    # Data rows start with \textbf{A}, \textbf{B}, etc.
    choices = []
    
    for row in rows:
        # Check if this is a data row (starts with \textbf{X} &)
        data_row_match = re.match(r'\\textbf\{([A-Z])\}\s*&\s*(.+)', row)
        if data_row_match:
            letter = data_row_match.group(1)
            cell_content = data_row_match.group(2)
            
            # Split the remaining content by & to get individual cell values
            cells = [c.strip() for c in cell_content.split('&')]
            
            # Join cells with comma separator for the choice string
            choice_str = ', '.join(cells)
            choices.append(choice_str)
    
    if not choices:
        print(f"  [!] Warning Q{q_num}: No data rows found in grid tabular")
    
    return choices


def find_duplicates(items: list[str]) -> list[tuple[int, int]]:
    """Find indices of duplicate items."""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                duplicates.append((i, j))
    return duplicates


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate detection."""
    # Remove all whitespace and common LaTeX spacing
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'\\[,;:!]', '', text)  # LaTeX spacing commands
    return text.lower()


def clean_latex_choice(text: str) -> str:
    """
    Clean a single choice text for JSON output.
    
    - Remove trailing \\ or \\[Xpt]
    - Normalize whitespace while preserving structure
    """
    # Remove trailing \\ or \\[Xpt]
    text = re.sub(r'\\\\\s*(\[\d+pt\])?\s*$', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize internal whitespace (but not aggressively)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text


def clean_latex_stem(text: str) -> str:
    """
    Clean LaTeX stem for JSON output.
    
    - Normalize whitespace while PRESERVING array row separators (\\)
    - Keep LaTeX math intact
    - Remove formatting artifacts
    """
    # First, protect array row separators by temporarily replacing them
    # Pattern: \\ followed by optional [Xpt] inside array environments
    # We want to keep \\ that are row separators in arrays
    
    # Find array environments and preserve their structure
    def preserve_array_structure(match):
        """Keep array content intact, just normalize whitespace within cells."""
        array_content = match.group(1)
        # Keep the structure, just normalize excessive whitespace
        # Replace multiple spaces/tabs with single space, but keep newlines as \\
        return '\\begin{array}' + array_content + '\\end{array}'
    
    # For arrays, we need to keep the \\ row separators
    # Let's handle this more carefully
    
    # Split by array environments
    parts = re.split(r'(\\begin\{array\}.*?\\end\{array\})', text, flags=re.DOTALL)
    
    cleaned_parts = []
    for part in parts:
        if '\\begin{array}' in part:
            # This is an array - preserve \\ as row separators
            # Just normalize spaces within lines, not the structure
            # Normalize excessive whitespace within the array but keep \\
            part = re.sub(r'[ \t]+', ' ', part)
            # Ensure \\ between rows is preserved (it already should be)
            cleaned_parts.append(part.strip())
        else:
            # Regular text - can normalize more aggressively
            part = re.sub(r'\s+', ' ', part)
            cleaned_parts.append(part.strip())
    
    text = ' '.join(p for p in cleaned_parts if p)
    
    # Remove \\[Xpt] line break commands (but not \\ inside arrays)
    # Only remove \\[Xpt] that are outside arrays (paragraph breaks)
    # This is tricky... let's be conservative and only remove at end
    text = re.sub(r'\\\\\[\d+pt\]\s*$', '', text)
    
    # Remove standalone \\ at end (paragraph break, not array separator)
    text = re.sub(r'\\\\$', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def merge_with_metadata(questions: list[dict], metadata: list[dict]) -> list[dict]:
    """
    Merge parsed questions with metadata from original corpus.
    
    Takes id, difficulty, and topics from metadata.
    All other fields come from parsed questions.
    """
    result = []
    
    for i, q in enumerate(questions):
        # Get metadata for this question (by position, not by question number)
        meta = metadata[i] if i < len(metadata) else {}
        
        # Build output with all fields from json template
        output = {
            "id": meta.get("id"),  # null if not present
            "difficulty": meta.get("difficulty"),  # null if not present
            "answer_key": q["answer_key"],
            "last_updated": date.today().isoformat(),  # current date in YYYY-MM-DD format
            "topics": meta.get("topics", []),  # empty list if not present
            "prompt": q["prompt"],
            "choices": q["choices"]
        }
        
        result.append(output)
    
    return result


def questions_to_jsonl(questions: list[dict]) -> str:
    """Convert questions to JSONL format."""
    lines = []
    
    for q in questions:
        json_line = json.dumps(q, ensure_ascii=False)
        lines.append(json_line)
    
    return '\n'.join(lines)


def validate_questions(questions: list[dict]) -> tuple[int, int]:
    """
    Validate parsed questions for common issues.
    
    Returns (warnings, errors) count.
    """
    warnings = 0
    errors = 0
    
    for i, q in enumerate(questions, 1):
        # Check for empty choices
        if not q.get('choices'):
            print(f"  [!] Error Q{i}: No choices found")
            errors += 1
            continue
        
        # Check for very short choices (possible truncation)
        for j, choice in enumerate(q['choices']):
            if len(choice) < 10 and not re.match(r'^[\$\d\.\s]+$', choice):
                # Very short and not just a number/math
                print(f"  [!] Warning Q{i} choice {chr(65+j)}: Very short ({len(choice)} chars): {choice[:50]}")
                warnings += 1
        
        # Check answer key is valid
        if q['answer_key'] >= len(q['choices']):
            print(f"  [!] Error Q{i}: Answer key {q['answer_key']} out of range (only {len(q['choices'])} choices)")
            errors += 1
    
    return warnings, errors


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    
    # Fixed file paths (no CLI arguments)
    input_tex = script_dir / "input_update.tex"
    metadata_jsonl = script_dir / "metadata_update.jsonl"
    output_jsonl = script_dir / "output_update.jsonl"
    
    # Clear output file first (so stale output doesn't persist if we error out)
    output_jsonl.write_text("", encoding='utf-8')
    
    # Check input file
    print(f"Reading LaTeX: {input_tex.name}")
    if not input_tex.exists():
        print(f"Error: Input file not found: {input_tex}")
        sys.exit(1)
    
    latex_content = input_tex.read_text(encoding='utf-8').strip()
    if not latex_content:
        print(f"Error: Input file is empty: {input_tex}")
        sys.exit(1)
    
    # Load optional metadata
    print(f"Reading metadata: {metadata_jsonl.name}", end="")
    metadata = load_metadata(metadata_jsonl)
    if metadata:
        print(f" ({len(metadata)} entries)")
    else:
        print(" (empty or not found - using null metadata)")
    
    # Parse questions from LaTeX
    print(f"Parsing LaTeX...")
    questions = extract_questions(latex_content)
    
    if not questions:
        print("Error: No questions found! Check the LaTeX format.")
        sys.exit(1)
    
    print(f"Found {len(questions)} questions")
    
    # Validate
    print(f"Validating...")
    warnings, errors = validate_questions(questions)
    if errors > 0:
        print(f"  Found {errors} errors and {warnings} warnings")
    elif warnings > 0:
        print(f"  Found {warnings} warnings (no errors)")
    else:
        print(f"  All questions validated OK")
    
    # Warn if metadata count doesn't match
    if metadata and len(metadata) != len(questions):
        print(f"  [!] Warning: Metadata has {len(metadata)} entries but LaTeX has {len(questions)} questions")
    
    # Merge with metadata
    merged = merge_with_metadata(questions, metadata)
    
    # Convert to JSONL
    jsonl_content = questions_to_jsonl(merged)
    
    # Write output
    output_jsonl.write_text(jsonl_content, encoding='utf-8')
    print(f"Wrote: {output_jsonl.name}")
    
    # Summary
    print(f"\n[OK] Successfully parsed {len(merged)} questions")
    for i, q in enumerate(merged):
        num_choices = len(q['choices'])
        has_id = "id" if q.get('id') else "no-id"
        has_topics = f"{len(q.get('topics', []))}t" if q.get('topics') else "0t"
        print(f"  Q{i+1}: {num_choices} choices, ans={q['answer_key']}, {has_id}, {has_topics}")


if __name__ == "__main__":
    main()
