import os
import json
import time
import random
import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from spinner import Spinner
from cost_report import calculate_cost, format_params, print_full_cost_report, print_summary_cost_report

# Initialize OpenAI API
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=API_KEY)

# Get script directory
SCRIPT_DIR = Path(__file__).parent
LEVEL_INSTRUCTIONS_FILE = SCRIPT_DIR.parent / "level_instructions.txt"

# Stage-specific configurations (each stage can use a different model)
GENERATOR_CONFIG = {
    "model": "gpt-5.2",  
    "reasoning_effort": "none",       # Required for `temperature` parameter in GPT5
    "max_completion_tokens": 20000,
    "temperature": 1.4
}
CORRECTOR_CONFIG = {
    "model": "gpt-5.2",
    "reasoning_effort": "high",
    "max_completion_tokens": 32000
}
VALIDATOR_CONFIG = {
    "model": "gpt-5.2",
    "reasoning_effort": "high",
    "max_completion_tokens": 16000
}

INPUT_FILE = SCRIPT_DIR / "0_input.jsonl"
PROMPT_FILE = SCRIPT_DIR / "prompt_gen.txt"
BLACKLIST_FILE = SCRIPT_DIR / "prompt_blacklist.txt"
CORRECTOR_PROMPT_FILE = SCRIPT_DIR / "prompt_corrector.txt"
CRITIC_PROMPT_FILE = SCRIPT_DIR / "prompt_critic.txt"
SPEC_FILE = SCRIPT_DIR / "tmua_spec.txt"

# Output files for each stage (numbered for natural sort order)
OUTPUT_GEN_JSON = SCRIPT_DIR / "1_gen.json"
OUTPUT_GEN_JSONL = SCRIPT_DIR / "1_gen.jsonl"
OUTPUT_CORRECTED_JSON = SCRIPT_DIR / "2_corrected.json"
OUTPUT_CORRECTED_JSONL = SCRIPT_DIR / "2_corrected.jsonl"
OUTPUT_VALIDATED_JSON = SCRIPT_DIR / "3_validated.json"
OUTPUT_VALIDATED_JSONL = SCRIPT_DIR / "3_validated.jsonl"

# Keys we never want to preserve in generated output if they carry no data
UNWANTED_METADATA_KEYS = {"exam", "year"}


def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_examples(input_file):
    """Load example questions from JSONL file, filtering out specified keys."""
    examples = []
    excluded_keys = {"id", "year", "exam", "difficulty"}
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                example = json.loads(line)
                # Remove excluded keys
                filtered_example = {k: v for k, v in example.items() if k not in excluded_keys}
                examples.append(filtered_example)
    
    # Randomly shuffle the examples so the model sees them in varied order
    random.shuffle(examples)
    
    return examples


def save_questions_dual(questions, json_path, jsonl_path):
    """Save questions to both JSON (pretty) and JSONL (one per line) formats."""
    # Save as pretty JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    # Save as JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")


def empty_output_files():
    """Empty all output files at the start of a run."""
    output_files = [
        OUTPUT_GEN_JSON,
        OUTPUT_GEN_JSONL,
        OUTPUT_CORRECTED_JSON,
        OUTPUT_CORRECTED_JSONL,
        OUTPUT_VALIDATED_JSON,
        OUTPUT_VALIDATED_JSONL,
    ]
    
    for file_path in output_files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            pass  # Empty the file


def generate_questions(examples, prompt_text, spec_text="", level_text="", special_prompt="", blacklist_text=""):
    """Generate novel questions using OpenAI API (Stage 1: Generator)."""
    start_time = time.time()
    
    # Format examples for the prompt
    examples_text = "\n\n".join([
        f"Example {i+1}:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
        for i, example in enumerate(examples)
    ])
    
    # Inject spec reference if provided
    if spec_text.strip():
        spec_section = f"\n\n**TMUA Content Specification (topics and knowledge requirements):**\n{spec_text}"
    else:
        spec_section = ""

    # Inject global level/notation instructions if provided
    level_section = ""
    if level_text and level_text.strip():
        level_section = f"\n\n**GLOBAL LEVEL & NOTATION INSTRUCTIONS (binding):**\n{level_text}"
    
    # Inject blacklist section if provided (specific questions to avoid regenerating)
    blacklist_section = ""
    if blacklist_text and blacklist_text.strip():
        blacklist_section = f"\n\n<<<BLACKLIST_START>>>\n**DO NOT GENERATE** questions identical or substantially similar to the following:\n{blacklist_text}\n<<<BLACKLIST_END>>>"
    
    # Inject special prompt section if provided (placed between main prompt and examples)
    special_prompt_section = ""
    if special_prompt and special_prompt.strip():
        special_prompt_section = f"\n\n**SPECIAL REQUEST (incorporate into generated questions without overriding example context):**\n{special_prompt}\n\n---===---"
    
    # Construct the full prompt
    full_prompt = f"{prompt_text}\n\n---===---{special_prompt_section}\n\nExisting example questions:\n\n{examples_text}{spec_section}{level_section}{blacklist_section}"
    
    print("\n--- Stage 1: Question Generation ---")
    print(f"Using {len(examples)} example(s) as context")
    
    # Call OpenAI API with spinner
    with Spinner("Generating questions"):
        response = client.chat.completions.create(
            model=GENERATOR_CONFIG["model"],
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            reasoning_effort=GENERATOR_CONFIG["reasoning_effort"],
            max_completion_tokens=GENERATOR_CONFIG["max_completion_tokens"],
            response_format={"type": "json_object"},
            temperature=GENERATOR_CONFIG["temperature"],
        )
    
    elapsed_time = time.time() - start_time
    
    # Extract the response
    result = response.choices[0].message.content
    
    # Get token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage'):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        print(f"✓ Generation complete ({elapsed_time:.1f}s)")
    
    return result, input_tokens, output_tokens, elapsed_time


def correct_questions(questions, corrector_prompt_text, spec_text="", level_text=""):
    """Correct generated questions using OpenAI API (Stage 2: Corrector)."""
    start_time = time.time()
    
    # Format questions as JSON input
    questions_input = {"questions": questions}
    questions_text = json.dumps(questions_input, indent=2, ensure_ascii=False)
    
    # Inject spec reference if provided
    if spec_text.strip():
        spec_section = f"\n\n**TMUA Content Specification (topics and knowledge requirements):**\n{spec_text}"
    else:
        spec_section = ""

    # Inject global level/notation instructions if provided
    level_section = ""
    if level_text and level_text.strip():
        level_section = f"\n\n**GLOBAL LEVEL & NOTATION INSTRUCTIONS (binding):**\n{level_text}"
    
    # Construct the full prompt
    full_prompt = f"{corrector_prompt_text}\n\nQuestions to correct:\n\n{questions_text}{spec_section}{level_section}"
    
    print("\n--- Stage 2: Question Correction ---")
    print(f"Correcting {len(questions)} question(s)")
    
    # Call OpenAI API with reasoning and spinner
    with Spinner("Correcting questions (with reasoning)"):
        response = client.chat.completions.create(
            model=CORRECTOR_CONFIG["model"],
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            reasoning_effort=CORRECTOR_CONFIG["reasoning_effort"],
            max_completion_tokens=CORRECTOR_CONFIG["max_completion_tokens"],
            response_format={"type": "json_object"},
        )
    
    elapsed_time = time.time() - start_time
    
    # Extract the response
    result = response.choices[0].message.content
    
    # Debug: Print the raw response if it looks empty or invalid
    if not result or not result.strip():
        print(f"  DEBUG: Empty response received from API")
        print(f"  DEBUG: Full response object: {response}")
        raise ValueError("Empty response from corrector API")
    
    # Get token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage'):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        print(f"✓ Correction complete ({elapsed_time:.1f}s)")
    
    return result, input_tokens, output_tokens, elapsed_time


def validate_question(question, critic_prompt_text, level_text=""):
    """Validate a single question using the critic prompt (Stage 3: Validator).
    
    Note: TMUA spec is intentionally NOT passed to the validator. Level-appropriateness
    is easy to manually check, massively increases input tokens, and causes edge case fails.
    """
    start_time = time.time()
    
    # Format the question for validation
    question_text = json.dumps(question, indent=2, ensure_ascii=False)

    # Inject global level/notation instructions if provided
    level_section = ""
    if level_text and level_text.strip():
        level_section = f"\n\n**GLOBAL LEVEL & NOTATION INSTRUCTIONS (binding):**\n{level_text}"
    
    # Construct the full prompt (no spec_text - level-appropriateness is manually checked)
    full_prompt = f"{critic_prompt_text}\n\nQuestion to evaluate:\n\n{question_text}{level_section}"
    
    # Call OpenAI API with spinner
    with Spinner("Validating"):
        response = client.chat.completions.create(
            model=VALIDATOR_CONFIG["model"],
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            reasoning_effort=VALIDATOR_CONFIG["reasoning_effort"],
            max_completion_tokens=VALIDATOR_CONFIG["max_completion_tokens"],
            response_format={"type": "json_object"},
        )
    
    elapsed_time = time.time() - start_time
    
    # Extract and parse the response
    result = response.choices[0].message.content
    
    # Debug: Print the raw response if it looks empty or invalid
    if not result or not result.strip():
        print(f"  DEBUG: Empty response received from API")
        print(f"  DEBUG: Full response object: {response}")
        raise ValueError("Empty response from API")
    
    # Get token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage'):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    
    try:
        evaluation = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"  DEBUG: Failed to parse JSON response")
        print(f"  DEBUG: Raw response content (first 500 chars): {result[:500]}")
        raise
    
    return evaluation, input_tokens, output_tokens, elapsed_time


def main():
    """Main function to generate, correct, and validate questions."""
    # Print configuration summary (without token rates)
    print("=" * 60)
    print("API CONFIG (per-stage model selection)")
    print(f"  Stage 1 Generator:  {GENERATOR_CONFIG['model']} | {format_params(GENERATOR_CONFIG)}")
    print(f"  Stage 2 Corrector:  {CORRECTOR_CONFIG['model']} | {format_params(CORRECTOR_CONFIG)}")
    print(f"  Stage 3 Validator:  {VALIDATOR_CONFIG['model']} | {format_params(VALIDATOR_CONFIG)}")
    print("=" * 60)
    
    # Start overall timing
    script_start_time = time.time()
    
    # Track all API calls for full cost report
    api_calls = []  # List of dicts with all call details
    
    # Track stage summaries for summary cost report
    stage_summary = {}  # {stage_name: {time, model, cost}}
    
    # Empty all output files at the start
    print("\nEmptying output files...")
    empty_output_files()
    
    print("\nLoading prompts and examples...")
    
    # Load prompts
    if not PROMPT_FILE.exists():
        print(f"Error: {PROMPT_FILE} not found")
        return
    prompt_text = load_text_file(PROMPT_FILE)
    
    if not CORRECTOR_PROMPT_FILE.exists():
        print(f"Error: {CORRECTOR_PROMPT_FILE} not found")
        return
    corrector_prompt_text = load_text_file(CORRECTOR_PROMPT_FILE)
    
    if not CRITIC_PROMPT_FILE.exists():
        print(f"Error: {CRITIC_PROMPT_FILE} not found")
        return
    critic_prompt_text = load_text_file(CRITIC_PROMPT_FILE)
    
    # Load TMUA spec (optional but recommended)
    spec_text = ""
    if SPEC_FILE.exists():
        spec_text = load_text_file(SPEC_FILE)
        if spec_text.strip():
            print(f"Loaded TMUA content specification from {SPEC_FILE}")
    else:
        print(f"Note: {SPEC_FILE} not found, proceeding without spec reference")

    # Load global level/notation instructions (optional but recommended)
    level_text = ""
    if LEVEL_INSTRUCTIONS_FILE.exists():
        level_text = load_text_file(LEVEL_INSTRUCTIONS_FILE)
        if level_text.strip():
            print(f"Loaded global level/notation instructions from {LEVEL_INSTRUCTIONS_FILE}")
        else:
            level_text = ""
            print(f"Note: {LEVEL_INSTRUCTIONS_FILE} is empty, proceeding without global level/notation instructions")
    else:
        print(f"Note: {LEVEL_INSTRUCTIONS_FILE} not found, proceeding without global level/notation instructions")

    # Load blacklist questions for generator (optional)
    blacklist_text = ""
    if BLACKLIST_FILE.exists():
        blacklist_text = load_text_file(BLACKLIST_FILE)
        if blacklist_text.strip():
            # Count JSON objects (lines starting with '{') for feedback
            question_count = sum(1 for l in blacklist_text.split('\n') if l.strip().startswith('{'))
            print(f"Loaded blacklist from {BLACKLIST_FILE} ({question_count} question(s) to avoid)")
        else:
            blacklist_text = ""
            print(f"Note: {BLACKLIST_FILE} is empty, proceeding without blacklist")
    else:
        print(f"Note: {BLACKLIST_FILE} not found, proceeding without blacklist")
    
    # Parse command-line arguments for special prompt
    special_prompt = ""
    if len(sys.argv) > 1:
        special_prompt = sys.argv[1]
        print(f"\nSpecial prompt provided: '{special_prompt}'")
    else:
        print("\nNo special prompt provided (using default behavior)")
    
    # Load examples from input.jsonl
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return
    
    examples = load_examples(INPUT_FILE)
    
    if not examples:
        print(f"Error: No examples found in {INPUT_FILE}")
        return
    
    print(f"Loaded {len(examples)} example question(s)")
    
    # ========== STAGE 1: GENERATION ==========
    try:
        result, input_tokens, output_tokens, elapsed_time = generate_questions(
            examples, prompt_text, spec_text, level_text, special_prompt, blacklist_text
        )
        
        cost = calculate_cost(GENERATOR_CONFIG["model"], input_tokens, output_tokens)
        
        # Track API call
        api_calls.append({
            "stage": "Generate",
            "api_call": "Generate-1",
            "model": GENERATOR_CONFIG["model"],
            "params": format_params(GENERATOR_CONFIG),
            "time": elapsed_time,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
        
        # Track stage summary
        stage_summary["Generate"] = {
            "time": elapsed_time,
            "model": GENERATOR_CONFIG["model"],
            "cost": cost
        }
        
        # Parse the JSON response
        response_obj = json.loads(result)
        
        # Expect response to have a "questions" array
        if "questions" not in response_obj:
            print("Error: Response does not contain 'questions' array")
            print(f"Response: {result}")
            return
        
        generated_questions = response_obj["questions"]

        # Strip unwanted metadata (e.g., null exam/year) before further processing
        for question in generated_questions:
            for key in list(question.keys()):
                if key in UNWANTED_METADATA_KEYS and question.get(key) is None:
                    question.pop(key, None)
        
        print(f"\n[OK] Successfully generated {len(generated_questions)} question(s)")
        
        # Save generated questions to gen.json and gen.jsonl
        save_questions_dual(generated_questions, OUTPUT_GEN_JSON, OUTPUT_GEN_JSONL)
        print(f"[OK] Saved to {OUTPUT_GEN_JSON} and {OUTPUT_GEN_JSONL}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response from generator: {e}")
        print(f"Response: {result}")
        return
    except Exception as e:
        print(f"Error during generation: {e}")
        return
    
    # ========== STAGE 2: CORRECTION ==========
    try:
        result, input_tokens, output_tokens, elapsed_time = correct_questions(
            generated_questions, corrector_prompt_text, spec_text, level_text
        )
        
        cost = calculate_cost(CORRECTOR_CONFIG["model"], input_tokens, output_tokens)
        
        # Track API call
        api_calls.append({
            "stage": "Correct",
            "api_call": "Correct-1",
            "model": CORRECTOR_CONFIG["model"],
            "params": format_params(CORRECTOR_CONFIG),
            "time": elapsed_time,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })
        
        # Track stage summary
        stage_summary["Correct"] = {
            "time": elapsed_time,
            "model": CORRECTOR_CONFIG["model"],
            "cost": cost
        }
        
        # Parse the JSON response
        response_obj = json.loads(result)
        
        # Expect response to have a "questions" array
        if "questions" not in response_obj:
            print("Error: Corrector response does not contain 'questions' array")
            print(f"Response: {result}")
            return
        
        corrected_questions = response_obj["questions"]

        # Strip unwanted metadata
        for question in corrected_questions:
            for key in list(question.keys()):
                if key in UNWANTED_METADATA_KEYS and question.get(key) is None:
                    question.pop(key, None)
        
        print(f"\n[OK] Successfully corrected {len(corrected_questions)} question(s)")
        
        # Save corrected questions to corrected.json and corrected.jsonl
        save_questions_dual(corrected_questions, OUTPUT_CORRECTED_JSON, OUTPUT_CORRECTED_JSONL)
        print(f"[OK] Saved to {OUTPUT_CORRECTED_JSON} and {OUTPUT_CORRECTED_JSONL}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response from corrector: {e}")
        print(f"Response: {result}")
        return
    except Exception as e:
        print(f"Error during correction: {e}")
        return
    
    # ========== STAGE 3: VALIDATION ==========
    print(f"\n--- Stage 3: Validation Pass ---")
    validated_questions = []
    validation_total_time = 0
    validation_total_cost = 0
    
    # Add 'valid' field as the first key in each question
    for i, question in enumerate(corrected_questions):
        corrected_questions[i] = {"valid": None, **question}
    
    for i, question in enumerate(corrected_questions, 1):
        print(f"\nValidating question {i}/{len(corrected_questions)}...")
        
        # Add a small delay to avoid rate limiting (except for the first question)
        if i > 1:
            time.sleep(1)
        
        try:
            evaluation, input_tokens, output_tokens, elapsed_time = validate_question(
                question, critic_prompt_text, level_text
            )
            
            cost = calculate_cost(VALIDATOR_CONFIG["model"], input_tokens, output_tokens)
            validation_total_time += elapsed_time
            validation_total_cost += cost
            
            # Track API call
            api_calls.append({
                "stage": "Validate",
                "api_call": f"Validate-{i}",
                "model": VALIDATOR_CONFIG["model"],
                "params": format_params(VALIDATOR_CONFIG),
                "time": elapsed_time,
                "cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
            
            well_posed = evaluation.get("well_posed", False)
            answer_correct = evaluation.get("answer_correct", False)
            overall_pass = evaluation.get("overall_pass", False)
            reasoning = evaluation.get("reasoning", "No reasoning provided")
            
            print(f"  Well-posed: {well_posed}, Answer correct: {answer_correct}, Overall: {'[PASSED]' if overall_pass else '[FAILED]'} ({elapsed_time:.1f}s)")
            
            # Update the valid field
            question["valid"] = overall_pass
            
            if not overall_pass:
                print(f"  Reasoning: {reasoning}")
            
            if overall_pass:
                validated_questions.append(question)
            
            # Update files in-place after each validation (only save passing questions)
            save_questions_dual(validated_questions, OUTPUT_VALIDATED_JSON, OUTPUT_VALIDATED_JSONL)
                
        except Exception as e:
            print(f"  [ERROR] Error during validation: {e}")
            print(f"  Question skipped")
            print(f"  DEBUG: Question that failed validation:")
            print(f"  {json.dumps(question, indent=2, ensure_ascii=False)[:300]}...")
    
    # Track validation stage summary
    stage_summary["Validate"] = {
        "time": validation_total_time,
        "model": VALIDATOR_CONFIG["model"],
        "cost": validation_total_cost
    }
    
    print(f"\n--- Validation Complete ---")
    print(f"Passed: {len(validated_questions)}/{len(corrected_questions)} questions")
    print(f"[OK] All questions saved to {OUTPUT_VALIDATED_JSON} and {OUTPUT_VALIDATED_JSONL}")
    
    if not validated_questions:
        print("\n[WARNING] No questions passed validation.")
    else:
        # Print summary
        print("\nValidated questions summary:")
        for i, q in enumerate(validated_questions, 1):
            topics = q.get("topics", [])
            difficulty = q.get("difficulty", "N/A")
            print(f"  {i}. Difficulty: {difficulty}, Topics: {', '.join(topics)}")
    
    # Calculate totals
    script_elapsed_time = time.time() - script_start_time
    total_cost = sum(call['cost'] for call in api_calls)
    
    # Print full cost report (detailed table of all API calls)
    print_full_cost_report(api_calls, total_cost)
    
    # Print summary cost report (compact table by stage) - this is the FINAL report before prompt
    print_summary_cost_report(stage_summary, script_elapsed_time, total_cost)
    
    # Ask if user wants to clear the input.jsonl file
    print("\n")
    clear_response = input("Do you want to clear the 0_input.jsonl file? (y/n): ").strip().lower()
    
    if clear_response == 'y':
        try:
            INPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(INPUT_FILE, 'w', encoding='utf-8') as f:
                pass
            print(f"✓ Cleared {INPUT_FILE}")
        except Exception as e:
            print(f"✗ Error clearing file: {e}")
    else:
        print("Input file was not cleared.")


if __name__ == "__main__":
    main()
