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

# GPT-5 Pricing (as of 2025)
INPUT_TOKEN_COST = 1.25 / 1_000_000  # $1.25 per million tokens
OUTPUT_TOKEN_COST = 10.00 / 1_000_000  # $10.00 per million tokens

# Get script directory
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "input.jsonl"
PROMPT_FILE = SCRIPT_DIR / "extract_prompt.txt"
EXAMPLES_DIR = SCRIPT_DIR / "examples"
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


def extract_latex_from_json(questions, prompt_text, examples, special_prompt=""):
    """
    Use OpenAI API to convert JSON questions to LaTeX format
    based on provided examples.
    """
    # Start timing
    start_time = time.time()
    
    # Format examples for the prompt
    examples_text = "\n\n".join([
        f"Example from {ex['filename']}:\n```latex\n{ex['content']}\n```"
        for ex in examples
    ])
    
    # Format questions for the prompt
    questions_text = json.dumps(questions, indent=2, ensure_ascii=False)
    
    # Inject special prompt if provided
    if special_prompt.strip():
        special_prompt_text = f"\n\n**SPECIAL INSTRUCTIONS:** {special_prompt}"
    else:
        special_prompt_text = ""
    
    # Construct the full prompt
    full_prompt = f"""{prompt_text}{special_prompt_text}

**EXAMPLE LaTeX DOCUMENTS (COMPLETE WITH PREAMBLES):**

{examples_text}

**QUESTIONS TO CONVERT (JSON format):**

```json
{questions_text}
```

IMPORTANT: Generate a COMPLETE, COMPILABLE LaTeX document (including \\documentclass, all \\usepackage statements, preamble configurations, \\begin{{document}}, the formatted questions, and \\end{{document}}). 

The output must be ready to save as a .tex file and compile immediately without errors.

Copy ALL package imports, custom commands, and configurations from the example documents.

**ANSWER KEYS:** Each question includes an "answer_key" field (0-indexed: 0=A, 1=B, 2=C, etc.). Include this as a LaTeX comment immediately after each \\item line in the format: % Answer: [index] (Letter: [A/B/C/...]).
"""
    
    print("\n--- Starting LaTeX Extraction ---")
    print(f"Using {len(examples)} example file(s) as style reference")
    print(f"Converting {len(questions)} question(s)")
    if special_prompt.strip():
        print(f"Special prompt: '{special_prompt}'")
    print("Sending request to OpenAI API...")
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        max_completion_tokens=20000,
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
    """Main function to extract and format LaTeX from JSON."""
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
    
    # Clear output file at the start
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    
    print("\nLoading prompt, examples, and input questions...")
    
    # Load prompt
    if not PROMPT_FILE.exists():
        print(f"Error: {PROMPT_FILE} not found")
        return
    prompt_text = load_text_file(PROMPT_FILE)
    
    # Load example .tex files
    if not EXAMPLES_DIR.exists():
        print(f"Error: {EXAMPLES_DIR} directory not found")
        return
    
    examples = load_example_tex_files(EXAMPLES_DIR)
    
    if not examples:
        print(f"Error: No example .tex files found in {EXAMPLES_DIR}")
        return
    
    print(f"Loaded {len(examples)} example LaTeX file(s)")
    
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
            questions, prompt_text, examples, special_prompt
        )
        
        # Track token usage
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Parse the JSON response
        response_obj = json.loads(result)
        
        # Expect response to have a "latex_document" string (complete document)
        if "latex_document" not in response_obj:
            print("Error: Response does not contain 'latex_document' field")
            print(f"Response keys: {list(response_obj.keys())}")
            
            # Try to save raw response for debugging
            debug_file = SCRIPT_DIR / "debug_response.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(response_obj, f, indent=2, ensure_ascii=False)
            print(f"Raw response saved to {debug_file}")
            return
        
        latex_output = response_obj["latex_document"]
        print(f"\n[OK] Successfully extracted complete LaTeX document")
        
        # Write to output file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(latex_output)
        
        print(f"[OK] Saved complete LaTeX document to {OUTPUT_FILE}")
        
        # Print preview
        print("\n--- Preview (first 500 characters) ---")
        print(latex_output[:500])
        if len(latex_output) > 500:
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
    
    # Calculate costs
    input_cost = total_input_tokens * INPUT_TOKEN_COST
    output_cost = total_output_tokens * OUTPUT_TOKEN_COST
    total_cost = input_cost + output_cost
    
    print(f"Input cost:  ${input_cost:.4f} (${INPUT_TOKEN_COST * 1_000_000:.2f}/M tokens)")
    print(f"Output cost: ${output_cost:.4f} (${OUTPUT_TOKEN_COST * 1_000_000:.2f}/M tokens)")
    print(f"Total cost:  ${total_cost:.4f}")
    print("="*60)
    
    # Ask if user wants to clear the input.jsonl file
    print("\n")
    clear_response = input("Do you want to clear the input.jsonl file? (y/n): ").strip().lower()
    
    if clear_response == 'y':
        try:
            if INPUT_FILE.exists():
                INPUT_FILE.unlink()
                print(f"✓ Cleared {INPUT_FILE}")
            else:
                print(f"File {INPUT_FILE} does not exist.")
        except Exception as e:
            print(f"✗ Error clearing file: {e}")
    else:
        print("Input file was not cleared.")


if __name__ == "__main__":
    main()

