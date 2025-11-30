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
PROMPT_FILE = SCRIPT_DIR / "prompt_gen.txt"
CRITIC_PROMPT_FILE = SCRIPT_DIR / "prompt_critic.txt"
LEVEL_PROMPT_FILE = SCRIPT_DIR / "prompt_level.txt"
SPEC_FILE = SCRIPT_DIR / "tmua_spec.txt"
OUTPUT_JSON = SCRIPT_DIR / "gen.json"
OUTPUT_JSONL = SCRIPT_DIR / "gen.jsonl"

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


def generate_questions(examples, prompt_text, level_prompt_text, spec_text="", special_prompt=""):
    """Generate novel questions using OpenAI API."""
    # Start timing
    start_time = time.time()
    
    # Format examples for the prompt
    examples_text = "\n\n".join([
        f"Example {i+1}:\n{json.dumps(example, indent=2, ensure_ascii=False)}"
        for i, example in enumerate(examples)
    ])
    
    # Inject level prompt if provided
    if level_prompt_text.strip():
        level_prompt_section = f"\n\n**Level Requirements:**\n{level_prompt_text}"
    else:
        level_prompt_section = ""
    
    # Inject special prompt if provided
    if special_prompt.strip():
        special_prompt_text = f"\n\n**SPECIAL INSTRUCTIONS (light guidance, prioritize examples):** {special_prompt}"
    else:
        special_prompt_text = ""
    
    # Inject spec reference if provided
    if spec_text.strip():
        spec_section = f"\n\n**TMUA Content Specification (topics and knowledge requirements):**\n{spec_text}"
    else:
        spec_section = ""
    
    # Construct the full prompt
    full_prompt = f"{prompt_text}{level_prompt_section}{special_prompt_text}\n\nExisting example questions:\n\n{examples_text}{spec_section}"
    
    print("\n--- Starting Question Generation ---")
    print(f"Using {len(examples)} example(s) as context")
    if special_prompt.strip():
        print(f"Special prompt: '{special_prompt}'")
    print("Sending request to OpenAI API...")
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        reasoning_effort="none",
        max_completion_tokens=20000,
        response_format={"type": "json_object"},
        temperature=1.2,
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
        print(f"✓ Generation complete")
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    return result, input_tokens, output_tokens


def validate_question(question, critic_prompt_text, level_prompt_text, spec_text=""):
    """Validate a question using the critic prompt."""
    # Start timing
    start_time = time.time()
    
    # Format the question for validation
    question_text = json.dumps(question, indent=2, ensure_ascii=False)
    
    # Inject level prompt if provided
    if level_prompt_text.strip():
        level_prompt_section = f"\n\n**Level Requirements to Check:**\n{level_prompt_text}"
    else:
        level_prompt_section = ""
    
    # Inject spec reference if provided
    if spec_text.strip():
        spec_section = f"\n\n**TMUA Content Specification (for level-appropriateness check):**\n{spec_text}"
    else:
        spec_section = ""
    
    # Construct the full prompt
    full_prompt = f"{critic_prompt_text}{level_prompt_section}\n\nQuestion to evaluate:\n\n{question_text}{spec_section}"
    
    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        reasoning_effort="high",
        max_completion_tokens=16000,  # Must be large enough for reasoning tokens + JSON output
        response_format={"type": "json_object"},
    )
    
    # Calculate elapsed time
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


def main():
    """Main function to generate questions."""
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
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        pass  # Empty the file
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        pass  # Empty the file
    
    print("\nLoading prompts and examples...")
    
    # Load prompt
    prompt_text = load_text_file(PROMPT_FILE)
    
    # Load critic prompt
    if not CRITIC_PROMPT_FILE.exists():
        print(f"Error: {CRITIC_PROMPT_FILE} not found")
        return
    critic_prompt_text = load_text_file(CRITIC_PROMPT_FILE)
    
    # Load level prompt (optional)
    level_prompt_text = ""
    if LEVEL_PROMPT_FILE.exists():
        level_prompt_text = load_text_file(LEVEL_PROMPT_FILE)
        if level_prompt_text.strip():
            print(f"Loaded level requirements from {LEVEL_PROMPT_FILE}")
    else:
        print(f"Note: {LEVEL_PROMPT_FILE} not found, proceeding without level restrictions")
    
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
    
    # Generate questions
    try:
        result, input_tokens, output_tokens = generate_questions(examples, prompt_text, level_prompt_text, spec_text, special_prompt)
        
        # Track token usage
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
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
                if (
                    key in UNWANTED_METADATA_KEYS
                    and question.get(key) is None
                ):
                    question.pop(key, None)
        print(f"\n[OK] Successfully generated {len(generated_questions)} question(s)")
        
        # Add 'valid' field as the first key in each question
        for i, question in enumerate(generated_questions):
            # Create new dict with 'valid' as first key
            generated_questions[i] = {"valid": None, **question}
        
        # Save generated questions to gen.json (pretty printed)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(generated_questions, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved {len(generated_questions)} question(s) to {OUTPUT_JSON} (awaiting validation)")
        
        # Save to gen.jsonl (one JSON per line)
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for question in generated_questions:
                f.write(json.dumps(question, ensure_ascii=False) + "\n")
        print(f"[OK] Saved to {OUTPUT_JSONL} (awaiting validation)")
        
        # Validate each question using the critic
        print(f"\n--- Starting Validation Pass ---")
        validated_questions = []
        
        for i, question in enumerate(generated_questions, 1):
            print(f"\nValidating question {i}/{len(generated_questions)}...")
            
            # Add a small delay to avoid rate limiting (except for the first question)
            if i > 1:
                time.sleep(1)
            
            try:
                evaluation, input_tokens, output_tokens = validate_question(question, critic_prompt_text, level_prompt_text, spec_text)
                
                # Track token usage
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                well_posed = evaluation.get("well_posed", False)
                answer_correct = evaluation.get("answer_correct", False)
                level_appropriate = evaluation.get("level_appropriate", True)  # Default True for backwards compat
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
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(generated_questions, f, indent=2, ensure_ascii=False)
                with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
                    for q in generated_questions:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                    
            except Exception as e:
                print(f"  [ERROR] Error during validation: {e}")
                print(f"  Question skipped")
                # Print the question that failed for debugging
                print(f"  DEBUG: Question that failed validation:")
                print(f"  {json.dumps(question, indent=2, ensure_ascii=False)[:300]}...")
        
        print(f"\n--- Validation Complete ---")
        print(f"Passed: {len(validated_questions)}/{len(generated_questions)} questions")
        print(f"[OK] All questions saved to {OUTPUT_JSON} and {OUTPUT_JSONL} with validation status")
        
        if not validated_questions:
            print("\n[WARNING] No questions passed validation.")
        else:
            # Print summary
            print("\nValidated questions summary:")
            for i, q in enumerate(validated_questions, 1):
                topics = q.get("topics", [])
                difficulty = q.get("difficulty", "N/A")
                print(f"  {i}. Difficulty: {difficulty}, Topics: {', '.join(topics)}")
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON response: {e}")
        print(f"Response: {result}")
    except Exception as e:
        print(f"Error during generation: {e}")
    
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

