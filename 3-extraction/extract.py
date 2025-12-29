import os
import json
import time
import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from spinner import Spinner

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

# Question separators (edit these to customize spacing between questions)
# Between odd and even questions (1→2, 3→4, etc.): horizontal rule separator
SEP_ODD_TO_EVEN = "\n\n\\vfill\n\\hrulefill\n\\vfill\n\n"
# Between even and odd questions (2→3, 4→5, etc.): page break with spacing
SEP_EVEN_TO_ODD = "\n\n\\vspace*{10pt}\n\\newpage\n\\vspace*{10pt}\n\n"


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


def get_model_pricing(model_name):
    """Get pricing for a specific model from the loaded pricing data."""
    if model_name in PRICING["models"]:
        p = PRICING["models"][model_name]
        return p["input_cost_per_million"] / 1_000_000, p["output_cost_per_million"] / 1_000_000
    raise ValueError(f"Unknown model: {model_name}. Add it to openai_pricing.json")


def calculate_cost(model, input_tokens, output_tokens):
    """Calculate cost for an API call."""
    input_rate, output_rate = get_model_pricing(model)
    return input_tokens * input_rate + output_tokens * output_rate


def format_params(config):
    """Format config params for display (reasoning_effort, temperature if non-default)."""
    parts = [f"r={config.get('reasoning_effort', 'N/A')}"]
    if 'temperature' in config and config['temperature'] != 1.0:
        parts.append(f"t={config['temperature']}")
    return ", ".join(parts)


def print_full_cost_report(api_calls, total_cost):
    """Print detailed ASCII table of all individual API calls."""
    if not api_calls:
        return
    
    print("\n" + "=" * 120)
    print("                                           FULL COST REPORT (All API Calls)")
    print("=" * 120)
    
    # Column headers
    headers = ["Stage", "API-Call", "Model", "Params", "Time", "Cost", "Cost%", "In-Tokens", "Out-Tokens"]
    
    # Calculate column widths
    col_widths = [12, 12, 12, 16, 8, 10, 7, 12, 12]
    
    # Print header row
    header_row = "│"
    for header, width in zip(headers, col_widths):
        header_row += f" {header:^{width}} │"
    
    separator = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
    top_border = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
    bottom_border = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
    
    print(top_border)
    print(header_row)
    print(separator)
    
    # Print data rows
    for call in api_calls:
        cost_pct = (call['cost'] / total_cost * 100) if total_cost > 0 else 0
        row = "│"
        row += f" {call['stage']:<{col_widths[0]}} │"
        row += f" {call['api_call']:<{col_widths[1]}} │"
        row += f" {call['model']:<{col_widths[2]}} │"
        row += f" {call['params']:<{col_widths[3]}} │"
        row += f" {call['time']:>{col_widths[4]}.1f}s │"
        row += f" ${call['cost']:>{col_widths[5]-1}.4f} │"
        row += f" {cost_pct:>{col_widths[6]-1}.1f}% │"
        row += f" {call['input_tokens']:>{col_widths[7]},} │"
        row += f" {call['output_tokens']:>{col_widths[8]},} │"
        print(row)
    
    print(bottom_border)


def print_summary_cost_report(stage_summary, total_time, total_cost):
    """Print compact summary table of costs by stage."""
    if not stage_summary:
        return
    
    print("\n" + "=" * 75)
    print("                         SUMMARY COST REPORT")
    print("=" * 75)
    
    # Column headers
    headers = ["Stage", "Time", "Model", "Cost", "Cost%"]
    col_widths = [14, 10, 14, 12, 8]
    
    # Print header row
    header_row = "│"
    for header, width in zip(headers, col_widths):
        header_row += f" {header:^{width}} │"
    
    separator = "├" + "┼".join(["─" * (w + 2) for w in col_widths]) + "┤"
    top_border = "┌" + "┬".join(["─" * (w + 2) for w in col_widths]) + "┐"
    bottom_border = "└" + "┴".join(["─" * (w + 2) for w in col_widths]) + "┘"
    
    print(top_border)
    print(header_row)
    print(separator)
    
    # Print data rows
    for stage_name, data in stage_summary.items():
        cost_pct = (data['cost'] / total_cost * 100) if total_cost > 0 else 0
        row = "│"
        row += f" {stage_name:<{col_widths[0]}} │"
        row += f" {data['time']:>{col_widths[1]-1}.1f}s │"
        row += f" {data['model']:<{col_widths[2]}} │"
        row += f" ${data['cost']:>{col_widths[3]-1}.4f} │"
        row += f" {cost_pct:>{col_widths[4]-1}.1f}% │"
        print(row)
    
    print(separator)
    
    # Print totals row
    total_row = "│"
    total_row += f" {'TOTAL':<{col_widths[0]}} │"
    total_row += f" {total_time:>{col_widths[1]-1}.1f}s │"
    total_row += f" {'':<{col_widths[2]}} │"
    total_row += f" ${total_cost:>{col_widths[3]-1}.4f} │"
    total_row += f" {'100.0':>{col_widths[4]-1}}% │"
    print(total_row)
    
    print(bottom_border)


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
    
    # Call OpenAI API with spinner
    with Spinner("Extracting LaTeX snippets"):
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
        print(f"✓ Extraction complete ({elapsed_time:.1f}s)")
    
    return result, input_tokens, output_tokens, elapsed_time


def main():
    """Main function to extract individual LaTeX question snippets from JSON questions.
    
    Outputs individual question snippets WITHOUT page layout elements.
    Page layout and spacing between questions will be handled separately.
    """
    # Print configuration summary
    print("=" * 50)
    print(f"API CONFIG: model={API_CONFIG['model']} | {format_params(API_CONFIG)}")
    print("=" * 50)
    
    # Start overall timing
    script_start_time = time.time()
    
    # Track all API calls for cost reporting
    api_calls = []
    
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
        result, input_tokens, output_tokens, elapsed_time = extract_latex_from_json(
            questions, prompt_text, style_prompt_text, examples, special_prompt
        )
        
        # Track API call
        cost = calculate_cost(API_CONFIG["model"], input_tokens, output_tokens)
        api_calls.append({
            "stage": "Extract",
            "api_call": "Extract-1",
            "model": API_CONFIG["model"],
            "params": format_params(API_CONFIG),
            "time": elapsed_time,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
        
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
    
    # Calculate totals
    script_elapsed_time = time.time() - script_start_time
    total_cost = sum(call['cost'] for call in api_calls)
    
    # Build stage summary (single stage for extract)
    stage_summary = {}
    if api_calls:
        stage_summary["Extract"] = {
            "time": sum(call['time'] for call in api_calls),
            "model": API_CONFIG["model"],
            "cost": total_cost
        }
    
    # Print full cost report (detailed table of all API calls)
    print_full_cost_report(api_calls, total_cost)
    
    # Print summary cost report (compact table by stage) - FINAL report before prompt
    print_summary_cost_report(stage_summary, script_elapsed_time, total_cost)
    
    # Ask if user wants to clear the input.jsonl file
    print("\n")
    clear_response = input("Do you want to clear the input_extract.jsonl file? (y/n): ").strip().lower()
    
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
