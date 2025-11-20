# 4-Production: Document Compilation

This layer takes structured question snippets from layer 3 and compiles them into a complete, properly formatted LaTeX document ready for compilation.

## Purpose

- Assembles questions from `3-extraction/output_raw.tex` into a full document
- Adds proper spacing between questions (2 per page)
- Injects questions into the preamble template
- **No LLM API calls required** - pure parsing and assembly

## Structure

```
4-production/
├── compile.py       # Main compilation script
├── preamble.tex     # LaTeX preamble and document structure
├── output.tex       # Generated complete document (output)
└── README.md        # This file
```

## Usage

### Basic Usage

```bash
python compile.py <num_questions>
```

This will:
- Read questions from `../3-extraction/output_raw.tex`
- Generate a document with the specified number of questions
- Output to `output.tex` in the current directory

### Advanced Usage

```bash
python compile.py <num_questions> [input_file] [output_file]
```

**Arguments:**
- `num_questions`: Number of questions to include (required)
- `input_file`: Path to input file with delimited questions (optional, default: `../3-extraction/output_raw.tex`)
- `output_file`: Path to output document (optional, default: `output.tex`)

### Examples

```bash
# Compile first 10 questions
python compile.py 10

# Compile 5 questions from custom input
python compile.py 5 custom_questions.tex

# Compile 8 questions to custom output location
python compile.py 8 ../3-extraction/output_raw.tex final_exam.tex
```

## Document Layout

The script creates a document with the following structure:

- **2 questions per page**
- **Between odd and even questions (Q1→Q2):** `\vfill \hrulefill \vfill` (horizontal rule in middle of page)
- **Between even and odd questions (Q2→Q3):** `\vspace{20pt} \newpage \vspace*{20pt}` (page break with spacing)
- **Document start:** `\vspace*{15pt}` (top margin)

## Input Format

The input file must contain questions delimited by:

```latex
% ===== QUESTION N =====
% Answer: [index] (Letter: [A/B/C/...])
\item
...question content...
% ===== END QUESTION N =====
```

The parser extracts the content between these delimiters and discards the delimiter comments.

## Output Format

Complete LaTeX document ready to compile:

```latex
% --- IMPORTS ---
\documentclass[leqno]{article}
...preamble content...

\begin{document}
\begin{enumerate}[label=\textbf{\arabic*.}, leftmargin=*, itemsep=0.75ex]

\vspace*{15pt}

% QUESTION 1
\item ...
% END QUESTION 1

\vfill
\hrulefill
\vfill

% QUESTION 2
\item ...
% END QUESTION 2

\vspace{20pt}
\newpage
\vspace*{20pt}

% QUESTION 3
...

\end{enumerate}
\end{document}
```

## Workflow Integration

This is the final step in the pipeline:

1. **1-transcription**: Convert images to JSON
2. **2-generation**: Generate MCQ questions
3. **3-extraction**: Convert JSON to LaTeX snippets → `output_raw.tex`
4. **4-production**: Compile snippets to complete document → `output.tex` ← **YOU ARE HERE**

After running this script, compile the output with:

```bash
pdflatex output.tex
```

Or your preferred LaTeX compiler.

## Notes

- This script does **not** require any API calls or credentials
- Questions are parsed using regex matching on delimiter comments
- The script will warn if you request more questions than available
- All spacing and layout is handled deterministically by this script
- The preamble file contains all LaTeX imports and document configuration

