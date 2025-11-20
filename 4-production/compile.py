"""
4-production/compile.py

Compiles a full LaTeX document from structured question snippets.
Takes structured input from layer 3 (output_raw.tex with delimited questions)
and produces a complete compilable document with proper spacing.

Usage:
    python compile.py <num_questions> [input_file] [output_file]
    
Arguments:
    num_questions: Number of questions to include in document
    input_file: (optional) Path to input file with question snippets 
                Default: ../3-extraction/output_raw.tex
    output_file: (optional) Path to output document
                 Default: output.tex
                 
Document layout:
    - 2 questions per page
    - Between odd and even (Q1→Q2): \vfill \hrulefill \vfill
    - Between even and odd (Q2→Q3): \vspace{20pt} \newpage \vspace*{20pt}
    - Start document with: \vspace*{15pt}
"""

import sys
import re
from pathlib import Path


# Get script directory
SCRIPT_DIR = Path(__file__).parent
PREAMBLE_FILE = SCRIPT_DIR / "preamble.tex"


def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_questions(input_text):
    """
    Parse question snippets from delimited text.
    
    Returns: List of question strings (without delimiters)
    """
    # Pattern to match: % ===== QUESTION N ===== ... % ===== END QUESTION N =====
    pattern = r'% ===== QUESTION \d+ =====\s*\n(.*?)\n% ===== END QUESTION \d+ ====='
    
    matches = re.findall(pattern, input_text, re.DOTALL)
    
    if not matches:
        print("Warning: No questions found with standard delimiters")
        return []
    
    # Clean up each question (strip leading/trailing whitespace)
    questions = [q.strip() for q in matches]
    
    return questions


def assemble_document(questions, num_questions):
    """
    Assemble questions into document body with proper spacing.
    
    Args:
        questions: List of question strings
        num_questions: Number of questions to include
        
    Returns: String with complete question section (without preamble)
    """
    if num_questions > len(questions):
        print(f"Warning: Requested {num_questions} questions but only {len(questions)} available")
        num_questions = len(questions)
    
    # Start document content
    parts = []
    parts.append(r"\vspace*{15pt}")
    parts.append("")
    
    for i in range(num_questions):
        # Add the question
        parts.append(f"% QUESTION {i+1}")
        parts.append("")
        parts.append(questions[i])
        parts.append("")
        parts.append(f"% END QUESTION {i+1}")
        
        # Add spacing based on position
        if i < num_questions - 1:  # Not the last question
            question_num = i + 1
            parts.append("")
            
            if question_num % 2 == 1:
                # Odd to even: middle of page
                parts.append(r"\vfill")
                parts.append(r"\hrulefill")
                parts.append(r"\vfill")
            else:
                # Even to odd: page break
                parts.append(r"\vspace{20pt}")
                parts.append(r"\newpage")
                parts.append(r"\vspace*{20pt}")
            
            parts.append("")
    
    return "\n".join(parts)


def compile_document(preamble_path, questions, num_questions, output_path):
    """
    Compile full LaTeX document.
    
    Args:
        preamble_path: Path to preamble.tex
        questions: List of question strings
        num_questions: Number of questions to include
        output_path: Path to output file
    """
    # Load preamble
    if not preamble_path.exists():
        raise FileNotFoundError(f"Preamble file not found: {preamble_path}")
    
    preamble_content = load_text_file(preamble_path)
    
    # Assemble question body
    questions_body = assemble_document(questions, num_questions)
    
    # Find injection point in preamble
    # Look for the comment marker or between \begin{enumerate} and \end{enumerate}
    if "% --- Output.tex goes here ---" in preamble_content:
        # Replace the comment with the actual content
        full_document = preamble_content.replace(
            "% --- Output.tex goes here ---",
            questions_body
        )
    else:
        # Fallback: inject between \begin{enumerate} and \end{enumerate}
        pattern = r'(\\begin\{enumerate\})(.*?)(\\end\{enumerate\})'
        full_document = re.sub(
            pattern,
            r'\1\n' + questions_body + r'\n\3',
            preamble_content,
            flags=re.DOTALL
        )
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_document)
    
    print(f"✓ Compiled document with {num_questions} question(s) to {output_path}")


def main():
    """Main function."""
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Error: Number of questions not provided")
        print()
        print("Usage: python compile.py <num_questions> [input_file] [output_file]")
        print()
        print("Arguments:")
        print("  num_questions: Number of questions to include")
        print("  input_file:    (optional) Path to input file")
        print("                 Default: ../3-extraction/output_raw.tex")
        print("  output_file:   (optional) Path to output file")
        print("                 Default: output.tex")
        sys.exit(1)
    
    # Get number of questions
    try:
        num_questions = int(sys.argv[1])
    except ValueError:
        print(f"Error: Invalid number of questions: {sys.argv[1]}")
        sys.exit(1)
    
    if num_questions <= 0:
        print(f"Error: Number of questions must be positive: {num_questions}")
        sys.exit(1)
    
    # Get input file path (default to layer 3 output)
    if len(sys.argv) >= 3:
        input_file = Path(sys.argv[2])
    else:
        input_file = SCRIPT_DIR.parent / "3-extraction" / "output_raw.tex"
    
    # Get output file path
    if len(sys.argv) >= 4:
        output_file = Path(sys.argv[3])
    else:
        output_file = SCRIPT_DIR / "output.tex"
    
    # Check input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"Reading questions from: {input_file}")
    print(f"Output document: {output_file}")
    print(f"Requested questions: {num_questions}")
    print()
    
    # Load and parse input
    input_text = load_text_file(input_file)
    questions = parse_questions(input_text)
    
    if not questions:
        print("Error: No questions found in input file")
        sys.exit(1)
    
    print(f"Found {len(questions)} question(s) in input file")
    
    # Compile document
    try:
        compile_document(PREAMBLE_FILE, questions, num_questions, output_file)
        print()
        print("✓ Compilation complete!")
    except Exception as e:
        print(f"Error during compilation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

