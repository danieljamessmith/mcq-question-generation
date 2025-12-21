import os
import json
import time
import sys
from pathlib import Path
from openai import OpenAI

# Initialize OpenAI API
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=API_KEY)

# Get script directory
SCRIPT_DIR = Path(__file__).parent
PRICING_FILE = SCRIPT_DIR.parent / "openai_pricing.json"
LEVEL_INSTRUCTIONS_FILE = SCRIPT_DIR.parent / "level_instructions.txt"

# Load pricing from external file
def load_pricing():
    """Load model pricing from external JSON file."""
    with open(PRICING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

PRICING = load_pricing()

# API Configuration
API_CONFIG = {
    "model": "gpt-5.2",
    "reasoning_effort": "high",
    "max_completion_tokens": 32000
}

INPUT_FILE = SCRIPT_DIR / "input_extract.jsonl"
PROMPT_FILE = SCRIPT_DIR / "prompt_extract.txt"
STYLE_PROMPT_FILE = SCRIPT_DIR / "prompt_style.txt"
EXAMPLES_DIR = SCRIPT_DIR / "examples"
PREAMBLE_FILE = SCRIPT_DIR / "preamble.tex"
OUTPUT_RAW_FILE = SCRIPT_DIR / "output_raw.tex"
OUTPUT_FILE = SCRIPT_DIR / "output.tex"


def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_jsonl_questions(input_file):
    """Load questions from JSONL file, filtering out metadata keys but KEEPING answer_key."""
    questions = []
    excluded_keys = {"id", "year", "exam", "difficulty"}  # Keep answer_key for LaTeX comments
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    question = json.loads(line)
                    # Remove excluded keys for cleaner input, but keep answer_key
                    filtered_question = {k: v for k, v in question.items() if k not in excluded_keys}
                    questions.append(filtered_question)
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_num}: {e}")
                    continue
    
    return questions


def load_example_tex_files(examples_dir):
    """Load all .tex files from examples directory."""
    tex_files = sorted(examples_dir.glob("*.tex"))
    
    if not tex_files:
        print(f"Warning: No .tex example files found in {examples_dir}")
        return []
    
    examples = []
    for tex_file in tex_files:
        content = load_text_file(tex_file)
        examples.append({
            "filename": tex_file.name,
            "content": content
        })
    
    return examples


def extract_latex_from_json(questions, prompt_text, style_prompt_text, examples, special_prompt=""):
    """
    Use OpenAI API to convert JSON questions to individual LaTeX snippets.
    Each snippet is self-contained and properly formatted, without page layout elements.
    Returns an array of question snippets delimited by LaTeX comments.
    """
    # Start timing
    start_time = time.time()
    
    # Format examples for the prompt
    examples_text = "\n\n".join([
        f"Example snippet from {ex['filename']}:\n```latex\n{ex['content']}\n```"
        for ex in examples
    ])
    
    # Format questions for the prompt
    questions_text = json.dumps(questions, indent=2, ensure_ascii=False)
    
    # Inject style prompt if provided
    if style_prompt_text.strip():
        style_prompt_section = f"\n\n**Style Requirements:**\n{style_prompt_text}"
    else:
        style_prompt_section = ""

    # Inject global level/notation instructions if provided
    level_section = ""
    if LEVEL_INSTRUCTIONS_FILE.exists():
        level_text = load_text_file(LEVEL_INSTRUCTIONS_FILE)
        if level_text.strip():
            level_section = f"\n\n**GLOBAL LEVEL & NOTATION INSTRUCTIONS (binding):**\n{level_text}"
    
    # Inject special prompt if provided
    if special_prompt.strip():
        special_prompt_text = f"\n\n**SPECIAL INSTRUCTIONS:** {special_prompt}"
    else:
        special_prompt_text = ""
    
    # Construct the full prompt
    full_prompt = f"""{prompt_text}{style_prompt_section}{level_section}{special_prompt_text}

**EXAMPLE LaTeX SNIPPETS:**

{examples_text}

**QUESTIONS TO CONVERT (JSON format):**

```json
{questions_text}
```

IMPORTANT: Generate INDIVIDUAL LaTeX SNIPPETS for each question. DO NOT include:
- Document structure: \\documentclass, \\usepackage, preamble, \\begin{{document}}/\\end{{document}}
- Page layout elements: \\hrulefill, \\vfill, \\vspace, \\newpage

Each snippet should:
1. Start with: % ===== QUESTION [N] =====
2. Include answer key comment: % Answer: [index] (Letter: [A/B/C/...])
3. Contain the formatted question content (\\item block with choices)
4. End with: % ===== END QUESTION [N] =====

The output should be an array of independent, self-contained question snippets only.
"""
    
    print("\n--- Starting LaTeX Snippet Extraction ---")
    print(f"Using {len(examples)} example file(s) as style reference")
    print(f"Converting {len(questions)} question(s) to individual LaTeX snippets")
    if special_prompt.strip():
        print(f"Special prompt: '{special_prompt}'")
    print("Sending request to OpenAI API...")
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model=API_CONFIG["model"],
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        reasoning_effort=API_CONFIG["reasoning_effort"],
        max_completion_tokens=API_CONFIG["max_completion_tokens"],
        response_format={"type": "json_object"},
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Extract the response
    result = response.choices[0].message.content
    
    # Get token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage'):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        print(f"✓ Extraction complete")
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    return result, input_tokens, output_tokens


def main():
    """Main function to extract individual LaTeX question snippets from JSON questions.
    
    Outputs individual question snippets WITHOUT page layout elements.
    Page layout and spacing between questions will be handled separately.
    """
    # Print configuration summary
    model_pricing = PRICING["models"][API_CONFIG["model"]]
    print("=" * 50)
    print(f"API CONFIG: model={API_CONFIG['model']} | reasoning={API_CONFIG['reasoning_effort']} | max_tokens={API_CONFIG['max_completion_tokens']}")
    print(f"PRICING: ${model_pricing['input_cost_per_million']:.2f}/M input, ${model_pricing['output_cost_per_million']:.2f}/M output")
    print(f"Pricing last updated: {PRICING.get('last_updated', 'unknown')}")
    print("=" * 50)
    
    # Start overall timing
    script_start_time = time.time()
    
    # Initialize token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Parse command-line arguments for special prompt
    special_prompt = ""
    if len(sys.argv) > 1:
        special_prompt = sys.argv[1]
        print(f"Special prompt provided: '{special_prompt}'")
    else:
        print("No special prompt provided (using default behavior)")
    
    # Clear output files at the start
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW_FILE, 'w', encoding='utf-8') as f:
        pass  # Empty the raw file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass  # Empty the final file
    
    print("\nLoading prompts, examples, and input questions...")
    
    # Load prompt
    if not PROMPT_FILE.exists():
        print(f"Error: {PROMPT_FILE} not found")
        return
    prompt_text = load_text_file(PROMPT_FILE)
    
    # Load style prompt (optional)
    style_prompt_text = ""
    if STYLE_PROMPT_FILE.exists():
        style_prompt_text = load_text_file(STYLE_PROMPT_FILE)
        if style_prompt_text.strip():
            print(f"Loaded style requirements from {STYLE_PROMPT_FILE}")
    else:
        print(f"Note: {STYLE_PROMPT_FILE} not found, proceeding without style requirements")
    
    # Load example .tex files (if any exist)
    # Note: Examples are optional - style guide is comprehensive
    examples = []
    if EXAMPLES_DIR.exists():
        examples = load_example_tex_files(EXAMPLES_DIR)
        if examples:
            print(f"Loaded {len(examples)} example LaTeX file(s)")
        else:
            print(f"Note: No example files found - using comprehensive style guide only")
    else:
        print(f"Note: No examples directory found - using comprehensive style guide only")
    
    # Load questions from input.jsonl
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return
    
    questions = load_jsonl_questions(INPUT_FILE)
    
    if not questions:
        print(f"Error: No questions found in {INPUT_FILE}")
        return
    
    print(f"Loaded {len(questions)} question(s)")
    
    # Extract and convert to LaTeX
    try:
        result, input_tokens, output_tokens = extract_latex_from_json(
            questions, prompt_text, style_prompt_text, examples, special_prompt
        )
        
        # Track token usage
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Parse the JSON response
        response_obj = json.loads(result)
        
        # Expect response to have a "latex_snippets" array
        if "latex_snippets" not in response_obj:
            print("Error: Response does not contain 'latex_snippets' field")
            print(f"Response keys: {list(response_obj.keys())}")
            
            # Try to save raw response for debugging
            debug_file = SCRIPT_DIR / "debug_response.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(response_obj, f, indent=2, ensure_ascii=False)
            print(f"Raw response saved to {debug_file}")
            return
        
        latex_snippets = response_obj["latex_snippets"]
        print(f"\n[OK] Successfully extracted {len(latex_snippets)} LaTeX snippet(s)")
        
        # Combine snippets with proper separators between them
        # Between odd and even questions (1→2, 3→4, etc.): \vfill \hrulefill \vfill
        # Between even and odd questions (2→3, 4→5, etc.): page break with spacing
        SEP_ODD_TO_EVEN = "\n\n\\vfill\n\\hrulefill\n\\vfill\n\n"
        SEP_EVEN_TO_ODD = "\n\n\\vspace*{10pt}\n\\newpage\n\\vspace*{10pt}\n\n"
        
        parts = []
        for i, snippet in enumerate(latex_snippets):
            parts.append(snippet)
            # Add separator after this snippet (if not the last one)
            if i < len(latex_snippets) - 1:
                question_num = i + 1  # 1-indexed question number
                if question_num % 2 == 1:
                    # After odd question (before even): horizontal rule separator
                    parts.append(SEP_ODD_TO_EVEN)
                else:
                    # After even question (before odd): page break
                    parts.append(SEP_EVEN_TO_ODD)
        
        latex_output = "".join(parts)
        
        # Write raw snippets to output_raw.tex
        with open(OUTPUT_RAW_FILE, "w", encoding="utf-8") as f:
            f.write(latex_output)
        
        print(f"[OK] Saved {len(latex_snippets)} snippet(s) to {OUTPUT_RAW_FILE}")
        
        # Generate full document by injecting snippets into preamble
        # Note: Page layout/spacing must be added manually or via a separate compilation step
        if PREAMBLE_FILE.exists():
            preamble_content = load_text_file(PREAMBLE_FILE)
            
            # Find the injection point (between \begin{enumerate} and \end{enumerate})
            # Look for the comment marker
            if "% --- Output.tex goes here ---" in preamble_content:
                # Replace the comment with the actual content
                full_document = preamble_content.replace(
                    "% --- Output.tex goes here ---",
                    latex_output
                )
            else:
                # Fallback: try to inject between \begin{enumerate} and \end{enumerate}
                import re
                pattern = r'(\\begin\{enumerate\})(.*?)(\\end\{enumerate\})'
                full_document = re.sub(
                    pattern,
                    r'\1\n' + latex_output + r'\n\3',
                    preamble_content,
                    flags=re.DOTALL
                )
            
            # Write full document to output.tex
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write(full_document)
            
            print(f"[OK] Generated full document at {OUTPUT_FILE}")
        else:
            print(f"[WARNING] {PREAMBLE_FILE} not found, only raw output generated")
        
        # Print preview
        print("\n--- Preview (first 800 characters) ---")
        print(latex_output[:800])
        if len(latex_output) > 800:
            print("...")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response: {e}")
        print(f"Response preview: {result[:500]}...")
        
        # Save raw response for debugging
        debug_file = SCRIPT_DIR / "debug_response.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Raw response saved to {debug_file}")
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
    
    # Calculate total elapsed time
    script_elapsed_time = time.time() - script_start_time
    
    # Print token usage and cost summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total elapsed time:  {script_elapsed_time:.2f}s")
    print("-"*60)
    
    # Calculate costs using pricing from JSON
    model_pricing = PRICING["models"][API_CONFIG["model"]]
    input_rate = model_pricing["input_cost_per_million"] / 1_000_000
    output_rate = model_pricing["output_cost_per_million"] / 1_000_000
    input_cost = total_input_tokens * input_rate
    output_cost = total_output_tokens * output_rate
    total_cost = input_cost + output_cost
    
    print(f"Model: {API_CONFIG['model']}")
    print(f"Input cost:  ${input_cost:.4f} (${model_pricing['input_cost_per_million']:.2f}/M tokens)")
    print(f"Output cost: ${output_cost:.4f} (${model_pricing['output_cost_per_million']:.2f}/M tokens)")
    print(f"Total cost:  ${total_cost:.4f}")
    print("="*60)
    
    # Ask if user wants to clear the input.jsonl file
    print("\n")
    clear_response = input("Do you want to clear the input.jsonl file? (y/n): ").strip().lower()
    
    if clear_response == 'y':
        try:
            # Create parent directory if it doesn't exist
            INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Empty the file (or create it if it doesn't exist)
            with open(INPUT_FILE, 'w', encoding='utf-8') as f:
                pass  # Write nothing to empty the file
            
            print(f"✓ Cleared {INPUT_FILE}")
        except Exception as e:
            print(f"✗ Error clearing file: {e}")
    else:
        print("Input file was not cleared.")


if __name__ == "__main__":
    main()

