# MCQ Question Generation Pipeline - Complete Architecture

## Overview

A 4-layer pipeline for generating professional LaTeX exam documents from handwritten question images.

```
[Images] → [1-Transcription] → [2-Generation] → [3-Extraction] → [4-Production] → [PDF]
```

---

## Layer 1: Transcription

**Purpose:** Convert handwritten question images to structured JSON

**Input:** Images in `1-transcription/img/`  
**Output:** `output.jsonl` (one JSON object per line)

**Process:**
- 🤖 Uses GPT-4-Vision API
- 📸 Reads handwritten question images
- 📝 Transcribes to structured JSON format
- ✅ Validates JSON structure

**Key Files:**
- `transcribe.py` - Main script
- `prompt_transcribe.txt` - Instructions for transcription
- `img/` - Input images directory
- `output.jsonl` - Transcribed questions

**Run:** `python transcribe.py`

---

## Layer 2: Generation

**Purpose:** Generate new MCQ questions using LLM

**Input:** Seed questions or prompts  
**Output:** `gen.jsonl` (generated questions in JSON format)

**Process:**
- 🤖 Uses GPT-5 API
- 💭 Generates questions based on patterns/topics
- 🎯 Applies difficulty levels
- ✅ Validates and critiques output

**Key Files:**
- `generate.py` - Main script
- `prompt_gen.txt` - Generation instructions
- `prompt_critic.txt` - Quality critique prompts
- `prompt_level.txt` - Difficulty level guidance
- `gen.jsonl` - Generated questions

**Run:** `python generate.py`

---

## Layer 3: Extraction

**Purpose:** Convert JSON questions to LaTeX snippets

**Input:** `input.jsonl` (JSON questions)  
**Output:** `output_raw.tex` (delimited LaTeX snippets)

**Process:**
- 🤖 Uses GPT-5 API
- 📐 Converts JSON to LaTeX format
- 🎨 Applies consistent styling
- 🔖 Adds delimiter comments for parsing
- ⚠️ **NO spacing commands** (pure content only)

**Key Files:**
- `extract.py` - Main script
- `prompt_extract.txt` - Extraction instructions
- `prompt_style.txt` - Style guidelines
- `examples/example1_raw.tex` - Style reference
- `input.jsonl` - Input questions
- `output_raw.tex` - **Delimited LaTeX snippets**

**Output Format:**
```latex
% ===== QUESTION 1 =====
% Answer: 0 (Letter: A)
\item
Question content here...
% ===== END QUESTION 1 =====

% ===== QUESTION 2 =====
% Answer: 2 (Letter: C)
\item
Question content here...
% ===== END QUESTION 2 =====
```

**Run:** `python extract.py`

---

## Layer 4: Production ⭐ NEW

**Purpose:** Compile delimited snippets into complete LaTeX document

**Input:** `3-extraction/output_raw.tex` (delimited snippets)  
**Output:** `output.tex` (complete compilable document)

**Process:**
- 🔍 Parse questions using regex on delimiters
- 📊 Select N questions as specified
- 📏 Inject spacing based on position:
  - Odd→Even (Q1→Q2): `\vfill \hrulefill \vfill`
  - Even→Odd (Q2→Q3): `\vspace{20pt} \newpage \vspace*{20pt}`
- 📄 Inject into preamble template
- ✅ Write complete document

**Key Features:**
- ⚡ **No LLM API calls** - deterministic parsing
- 🎯 **Compile any subset** of available questions
- 📐 **Consistent layout** - 2 questions per page
- 🔒 **Robust parsing** - delimiter-based extraction

**Key Files:**
- `compile.py` - Main compilation script
- `preamble.tex` - LaTeX document template
- `README.md` - Usage documentation
- `SETUP_SUMMARY.md` - Architecture details
- `output.tex` - Generated document

**Run:** `python compile.py <num_questions>`

**Example:**
```bash
python compile.py 10              # First 10 questions
python compile.py 5 input.tex     # Custom input
python compile.py 8 input.tex output.tex  # Custom input/output
```

---

## Complete Workflow

### Option A: From Images to PDF

```bash
# Step 1: Transcribe images
cd 1-transcription
# Add images to img/ directory
python transcribe.py

# Step 2: Generate more questions (optional)
cd ../2-generation
# Copy questions from layer 1 to input.jsonl if needed
python generate.py

# Step 3: Extract to LaTeX
cd ../3-extraction
# Copy questions to input.jsonl
python extract.py

# Step 4: Compile document
cd ../4-production
python compile.py 20

# Step 5: Generate PDF
pdflatex output.tex
```

### Option B: Direct JSON to PDF

```bash
# If you already have questions in JSON format:

# Step 1: Place in layer 3
cp questions.jsonl 3-extraction/input.jsonl

# Step 2: Extract to LaTeX
cd 3-extraction
python extract.py

# Step 3: Compile document
cd ../4-production
python compile.py 15

# Step 4: Generate PDF
pdflatex output.tex
```

---

## Data Flow Diagram

```
┌─────────────────┐
│  Images (.jpg)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  1-Transcription (GPT-4V)│
└────────┬────────────────┘
         │ output.jsonl
         ▼
┌─────────────────────────┐
│  2-Generation (GPT-5)    │◄─── Optional: Generate more
└────────┬────────────────┘
         │ gen.jsonl
         ▼
┌─────────────────────────┐
│  3-Extraction (GPT-5)    │
└────────┬────────────────┘
         │ output_raw.tex (delimited snippets)
         ▼
┌─────────────────────────┐
│  4-Production (No LLM)   │◄─── Deterministic parsing
└────────┬────────────────┘
         │ output.tex (complete document)
         ▼
┌─────────────────────────┐
│  pdflatex                │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  output.pdf     │ ✓ Final exam
└─────────────────┘
```

---

## Key Design Decisions

### ✅ Delimiter-Based Storage (Layer 3 → 4)

**Why delimiters?**
- ✓ Easy to parse without LLM
- ✓ Preserves exact LaTeX formatting
- ✓ Resistant to LLM formatting quirks
- ✓ Human-readable and editable

**Format:**
```latex
% ===== QUESTION N =====
...content...
% ===== END QUESTION N =====
```

**Alternative considered:** JSON with LaTeX strings
- ✗ Requires escaping LaTeX special characters
- ✗ Harder to manually edit
- ✓ Easier programmatic manipulation

**Decision:** Delimiters in plain `.tex` file for maintainability

### ✅ Separation of Content and Layout

**Layer 3 (Content):**
- Pure question content
- No spacing commands
- No page layout

**Layer 4 (Layout):**
- Deterministic spacing injection
- Page layout rules
- Document structure

**Benefits:**
- Change layout without re-running expensive LLM calls
- Compile different subsets from same content
- Debug content issues separately from layout issues

### ✅ Two-Stage Compilation

**Stage 1 (Layer 3 with LLM):**
```
JSON → LaTeX snippets
```
- Expensive (API calls)
- Variable output
- Run once per question set

**Stage 2 (Layer 4 without LLM):**
```
LaTeX snippets → Complete document
```
- Free (no API)
- Deterministic output
- Run many times as needed

**Benefits:**
- Iterate on layout without cost
- Generate multiple versions (10-question quiz, 20-question exam)
- Fast compilation (< 1 second)

---

## Document Layout Specification

### Spacing Pattern

**Page 1:**
```
\vspace*{15pt}           ← Top margin

QUESTION 1

\vfill                   ← Flexible vertical space
\hrulefill               ← Horizontal rule
\vfill                   ← Flexible vertical space

QUESTION 2

\vspace{20pt}            ← Fixed space before break
```

**Page 2:**
```
\newpage
\vspace*{20pt}           ← Top margin after break

QUESTION 3

\vfill
\hrulefill
\vfill

QUESTION 4

\vspace{20pt}
```

**Pattern continues:** Always 2 questions per page

---

## File Structure

```
mcq-question-generation/
├── 1-transcription/
│   ├── img/                    # Input images
│   ├── transcribe.py
│   ├── prompt_transcribe.txt
│   └── output.jsonl           # → feeds layer 2 or 3
│
├── 2-generation/
│   ├── generate.py
│   ├── prompt_gen.txt
│   ├── prompt_critic.txt
│   ├── prompt_level.txt
│   ├── input.jsonl            # Optional: seed questions
│   └── gen.jsonl              # → feeds layer 3
│
├── 3-extraction/
│   ├── examples/
│   │   └── example1_raw.tex
│   ├── extract.py
│   ├── prompt_extract.txt
│   ├── prompt_style.txt
│   ├── input.jsonl            # Input questions
│   └── output_raw.tex         # → feeds layer 4 ⭐
│
├── 4-production/              # ⭐ NEW LAYER
│   ├── compile.py             # Main script
│   ├── preamble.tex           # Document template
│   ├── README.md              # Usage guide
│   ├── SETUP_SUMMARY.md       # Architecture details
│   └── output.tex             # → compile to PDF
│
├── clear.py                   # Cleanup utility
├── json_validator.py          # Validation utility
├── requirements.txt           # Python dependencies
└── PIPELINE_ARCHITECTURE.md   # This file
```

---

## API Usage Summary

| Layer | API | Cost | Required For |
|-------|-----|------|--------------|
| 1 - Transcription | GPT-4-Vision | $$$ | Image → JSON |
| 2 - Generation | GPT-5 | $$ | Create questions |
| 3 - Extraction | GPT-5 | $$ | JSON → LaTeX |
| **4 - Production** | **None** | **Free** | **Compile document** |

**Total for 10 questions:** ~$0.50-1.00 (layers 1-3)  
**Recompiling with different layouts:** $0.00 (layer 4 only)

---

## Advantages of This Architecture

### 🎯 Modularity
- Each layer has a single responsibility
- Can skip layers (e.g., start from JSON)
- Easy to modify individual components

### 💰 Cost Efficiency
- Layer 4 compilation is free
- Generate once, compile many times
- No wasted API calls for layout tweaks

### 🔒 Reliability
- Deterministic compilation in layer 4
- Clear delimiters prevent parsing errors
- LLM errors isolated to content generation

### 🚀 Flexibility
- Compile any subset of questions
- Multiple output formats from same snippets
- Easy to add custom layouts

### 📝 Maintainability
- Human-readable intermediate files
- Can manually edit snippets if needed
- Clear data flow between layers

---

## Future Enhancements

### Potential Additions

1. **Answer Key Generation** (Layer 4)
   - Parse answer comments from snippets
   - Generate separate answer key document

2. **Question Bank Management** (New Layer)
   - Database of questions
   - Tag/categorize by topic
   - Select questions by criteria

3. **Multiple Layouts** (Layer 4 variants)
   - Quiz format (4 per page)
   - Exam format (current: 2 per page)
   - Practice sheet (no answer comments)

4. **Difficulty Balancing** (Layer 4)
   - Parse difficulty from metadata
   - Select balanced question sets
   - Group by difficulty level

5. **LaTeX Template Options** (Layer 4)
   - Multiple preamble templates
   - Different page layouts
   - Custom styling options

---

## Conclusion

The 4-layer architecture provides a robust, cost-effective pipeline for generating professional exam documents. The addition of Layer 4 (Production) separates content generation from document assembly, enabling:

- **Free, deterministic compilation**
- **Flexible question selection**
- **Consistent, professional layout**
- **Easy iteration and testing**

The system is production-ready and scalable to large question banks.

