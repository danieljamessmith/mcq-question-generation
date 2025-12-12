import os
import json
import time
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

# Load pricing from external file
def load_pricing():
    """Load model pricing from external JSON file."""
    with open(PRICING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

PRICING = load_pricing()

# Stage-specific configurations (each stage can use a different model)
GENERATOR_CONFIG = {
    "model": "gpt-5.1",  # Creative generation - 5.1 is sufficient
    "reasoning_effort": "none",
    "max_completion_tokens": 20000,
    "temperature": 1.4
}
CORRECTOR_CONFIG = {
    "model": "gpt-5.2",  # Reasoning-heavy - benefits from 5.2
    "reasoning_effort": "medium",
    "max_completion_tokens": 32000
}
VALIDATOR_CONFIG = {
    "model": "gpt-5.2",  # Reasoning-heavy - benefits from 5.2
    "reasoning_effort": "medium",
    "max_completion_tokens": 16000
}

INPUT_FILE = SCRIPT_DIR / "0_input.jsonl"
PROMPT_FILE = SCRIPT_DIR / "prompt_gen.txt"
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
    excluded_keys = {"id", "year", "exam"}
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                example = json.loads(line)
                # Remove excluded keys
                filtered_example = {k: v for k, v in example.items() if k not in excluded_keys}
                examples.append(filtered_example)
    
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


def generate_questions(examples, prompt_text, spec_text=""):
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
    
    # Construct the full prompt
    full_prompt = f"{prompt_text}\n\nExisting example questions:\n\n{examples_text}{spec_section}"
    
    print("\n--- Stage 1: Question Generation ---")
    print(f"Using {len(examples)} example(s) as context")
    print("Sending request to OpenAI API...")
    
    # Call OpenAI API
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
        print(f"✓ Generation complete")
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    return result, input_tokens, output_tokens


def correct_questions(questions, corrector_prompt_text, spec_text=""):
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
    
    # Construct the full prompt
    full_prompt = f"{corrector_prompt_text}\n\nQuestions to correct:\n\n{questions_text}{spec_section}"
    
    print("\n--- Stage 2: Question Correction ---")
    print(f"Correcting {len(questions)} question(s)")
    print("Sending request to OpenAI API (with reasoning)...")
    
    # Call OpenAI API with reasoning
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
        print(f"✓ Correction complete")
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    return result, input_tokens, output_tokens


def validate_question(question, critic_prompt_text, spec_text=""):
    """Validate a single question using the critic prompt (Stage 3: Validator)."""
    start_time = time.time()
    
    # Format the question for validation
    question_text = json.dumps(question, indent=2, ensure_ascii=False)
    
    # Inject spec reference if provided
    if spec_text.strip():
        spec_section = f"\n\n**TMUA Content Specification (for level-appropriateness check):**\n{spec_text}"
    else:
        spec_section = ""
    
    # Construct the full prompt
    full_prompt = f"{critic_prompt_text}\n\nQuestion to evaluate:\n\n{question_text}{spec_section}"
    
    # Call OpenAI API
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
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    try:
        evaluation = json.loads(result)
    except json.JSONDecodeError as e:
        print(f"  DEBUG: Failed to parse JSON response")
        print(f"  DEBUG: Raw response content (first 500 chars): {result[:500]}")
        raise
    
    return evaluation, input_tokens, output_tokens


def get_model_pricing(model_name):
    """Get pricing for a specific model from the loaded pricing data."""
    if model_name in PRICING["models"]:
        p = PRICING["models"][model_name]
        return p["input_cost_per_million"] / 1_000_000, p["output_cost_per_million"] / 1_000_000
    raise ValueError(f"Unknown model: {model_name}. Add it to openai_pricing.json")


def main():
    """Main function to generate, correct, and validate questions."""
    # Print configuration summary
    print("=" * 60)
    print("API CONFIG (per-stage model selection)")
    gen_p = PRICING["models"][GENERATOR_CONFIG["model"]]
    cor_p = PRICING["models"][CORRECTOR_CONFIG["model"]]
    val_p = PRICING["models"][VALIDATOR_CONFIG["model"]]
    print(f"  Stage 1 Generator:  {GENERATOR_CONFIG['model']} | reasoning={GENERATOR_CONFIG['reasoning_effort']:6} | max_tokens={GENERATOR_CONFIG['max_completion_tokens']} | temp={GENERATOR_CONFIG['temperature']}")
    print(f"                      Pricing: ${gen_p['input_cost_per_million']:.2f}/M in, ${gen_p['output_cost_per_million']:.2f}/M out")
    print(f"  Stage 2 Corrector:  {CORRECTOR_CONFIG['model']} | reasoning={CORRECTOR_CONFIG['reasoning_effort']:6} | max_tokens={CORRECTOR_CONFIG['max_completion_tokens']}")
    print(f"                      Pricing: ${cor_p['input_cost_per_million']:.2f}/M in, ${cor_p['output_cost_per_million']:.2f}/M out")
    print(f"  Stage 3 Validator:  {VALIDATOR_CONFIG['model']} | reasoning={VALIDATOR_CONFIG['reasoning_effort']:6} | max_tokens={VALIDATOR_CONFIG['max_completion_tokens']}")
    print(f"                      Pricing: ${val_p['input_cost_per_million']:.2f}/M in, ${val_p['output_cost_per_million']:.2f}/M out")
    print(f"Pricing last updated: {PRICING.get('last_updated', 'unknown')}")
    print("=" * 60)
    
    # Start overall timing
    script_start_time = time.time()
    
    # Initialize per-model token tracking
    token_usage = {}  # {model_name: {"input": count, "output": count}}
    
    # Empty all output files at the start
    print("\nEmptying output files...")
    empty_output_files()
    
    print("\nLoading prompts and examples...")
    
    # Load prompts
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
    
    # Load examples from input.jsonl
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return
    
    examples = load_examples(INPUT_FILE)
    
    if not examples:
        print(f"Error: No examples found in {INPUT_FILE}")
        return
    
    print(f"Loaded {len(examples)} example question(s)")
    
    # Helper to track tokens per model
    def track_tokens(model, input_tok, output_tok):
        if model not in token_usage:
            token_usage[model] = {"input": 0, "output": 0}
        token_usage[model]["input"] += input_tok
        token_usage[model]["output"] += output_tok
    
    # ========== STAGE 1: GENERATION ==========
    try:
        result, input_tokens, output_tokens = generate_questions(examples, prompt_text, spec_text)
        
        track_tokens(GENERATOR_CONFIG["model"], input_tokens, output_tokens)
        
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
        result, input_tokens, output_tokens = correct_questions(
            generated_questions, corrector_prompt_text, spec_text
        )
        
        track_tokens(CORRECTOR_CONFIG["model"], input_tokens, output_tokens)
        
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
    
    # Add 'valid' field as the first key in each question
    for i, question in enumerate(corrected_questions):
        corrected_questions[i] = {"valid": None, **question}
    
    for i, question in enumerate(corrected_questions, 1):
        print(f"\nValidating question {i}/{len(corrected_questions)}...")
        
        # Add a small delay to avoid rate limiting (except for the first question)
        if i > 1:
            time.sleep(1)
        
        try:
            evaluation, input_tokens, output_tokens = validate_question(
                question, critic_prompt_text, spec_text
            )
            
            track_tokens(VALIDATOR_CONFIG["model"], input_tokens, output_tokens)
            
            well_posed = evaluation.get("well_posed", False)
            answer_correct = evaluation.get("answer_correct", False)
            level_appropriate = evaluation.get("level_appropriate", True)
            overall_pass = evaluation.get("overall_pass", False)
            reasoning = evaluation.get("reasoning", "No reasoning provided")
            
            print(f"  Well-posed: {well_posed}")
            print(f"  Answer correct: {answer_correct}")
            print(f"  Level appropriate: {level_appropriate}")
            print(f"  Overall: {'[PASSED]' if overall_pass else '[FAILED]'}")
            
            # Update the valid field
            question["valid"] = overall_pass
            
            if not overall_pass:
                print(f"  Full validation reasoning:")
                print(f"  {reasoning}")
            
            if overall_pass:
                validated_questions.append(question)
            
            # Update files in-place after each validation
            save_questions_dual(corrected_questions, OUTPUT_VALIDATED_JSON, OUTPUT_VALIDATED_JSONL)
                
        except Exception as e:
            print(f"  [ERROR] Error during validation: {e}")
            print(f"  Question skipped")
            print(f"  DEBUG: Question that failed validation:")
            print(f"  {json.dumps(question, indent=2, ensure_ascii=False)[:300]}...")
    
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
    
    # Calculate total elapsed time
    script_elapsed_time = time.time() - script_start_time
    
    # Print token usage and cost summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate totals and per-model costs
    total_input_tokens = sum(u["input"] for u in token_usage.values())
    total_output_tokens = sum(u["output"] for u in token_usage.values())
    
    print(f"Total input tokens:  {total_input_tokens:,}")
    print(f"Total output tokens: {total_output_tokens:,}")
    print(f"Total tokens:        {total_input_tokens + total_output_tokens:,}")
    print(f"Total elapsed time:  {script_elapsed_time:.2f}s")
    print("-"*60)
    
    # Calculate per-model costs
    total_cost = 0
    for model, usage in sorted(token_usage.items()):
        input_rate, output_rate = get_model_pricing(model)
        model_input_cost = usage["input"] * input_rate
        model_output_cost = usage["output"] * output_rate
        model_total = model_input_cost + model_output_cost
        total_cost += model_total
        print(f"{model}: {usage['input']:,} in + {usage['output']:,} out = ${model_total:.4f}")
    
    print("-"*60)
    print(f"Total cost:  ${total_cost:.4f}")
    print("="*60)
    
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
