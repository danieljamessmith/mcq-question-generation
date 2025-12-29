import os
import json
import base64
import time
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

# API Configuration
API_CONFIG = {
    "model": "gpt-5.2",
    "reasoning_effort": "medium",
    "max_completion_tokens": 16000
}
KEYS_DIR = SCRIPT_DIR / "keys"
INPUT_FILE = SCRIPT_DIR / "output.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "output.jsonl"  # Overwrites the input
PROMPT_FILE = SCRIPT_DIR / "prompt_keys.txt"


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_questions(input_file):
    """Load questions from JSONL file."""
    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def get_image_media_type(image_path):
    """Determine the media type based on file extension."""
    ext = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/png")


def fill_answer_keys(questions, image_path, prompt_text):
    """Send questions and answer key image to OpenAI API to fill answer_keys."""
    # Start timing
    start_time = time.time()
    
    # Encode the image
    base64_image = encode_image(image_path)
    media_type = get_image_media_type(image_path)
    
    # Format questions as JSONL string for the prompt
    questions_jsonl = "\n".join([json.dumps(q, ensure_ascii=False) for q in questions])
    
    # Construct the full prompt
    full_prompt = f"{prompt_text}\n\n---===---\n\nINPUT JSONL:\n\n{questions_jsonl}"
    
    # Call OpenAI Vision API with spinner
    with Spinner("Filling answer keys"):
        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            reasoning_effort=API_CONFIG["reasoning_effort"],
            max_completion_tokens=API_CONFIG["max_completion_tokens"],
            response_format={"type": "json_object"},
        )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Extract the response and include usage info
    result = response.choices[0].message.content
    
    # Get token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage'):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
    
    print(f"✓ Response received ({elapsed_time:.1f}s)")
    
    return result, input_tokens, output_tokens, elapsed_time


def main():
    """Main function to fill answer keys from answer key images."""
    # Print configuration summary
    print("=" * 50)
    print(f"API CONFIG: model={API_CONFIG['model']} | {format_params(API_CONFIG)}")
    print("=" * 50)
    
    # Start overall timing
    script_start_time = time.time()
    
    # Track all API calls for cost reporting
    api_calls = []
    
    # Load prompt
    print("Loading prompt...")
    if not PROMPT_FILE.exists():
        print(f"Error: {PROMPT_FILE} not found")
        return
    prompt_text = load_text_file(PROMPT_FILE)
    
    # Load questions from input file
    print("Loading questions...")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return
    
    questions = load_questions(INPUT_FILE)
    
    if not questions:
        print(f"Error: No questions found in {INPUT_FILE}")
        return
    
    print(f"Loaded {len(questions)} question(s)")
    
    # Count questions with null answer_key
    null_keys_count = sum(1 for q in questions if q.get("answer_key") is None)
    print(f"Questions with null answer_key: {null_keys_count}")
    
    # Get all image files from keys directory (sorted by creation date, oldest first)
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    image_files = sorted(
        [f for f in KEYS_DIR.glob("*") if f.suffix.lower() in image_extensions],
        key=lambda x: x.stat().st_ctime
    )
    
    if not image_files:
        print(f"No image files found in {KEYS_DIR}")
        return
    
    print(f"Found {len(image_files)} answer key image(s) to process")
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        print(f"\nProcessing [{idx}/{len(image_files)}]: {image_path.name}")
        
        # Retry logic for failed attempts
        max_retries = 2
        result = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt}/{max_retries}...")
                    time.sleep(2)  # Brief delay before retry
                
                # Fill answer keys using the image
                result, input_tokens, output_tokens, elapsed_time = fill_answer_keys(
                    questions, image_path, prompt_text
                )
                
                # Track API call
                cost = calculate_cost(API_CONFIG["model"], input_tokens, output_tokens)
                api_calls.append({
                    "stage": "FillKeys",
                    "api_call": f"FillKeys-{idx}",
                    "model": API_CONFIG["model"],
                    "params": format_params(API_CONFIG),
                    "time": elapsed_time,
                    "cost": cost,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                })
                
                # Check for empty response
                if not result or not result.strip():
                    if attempt < max_retries:
                        print(f"  Empty response, retrying...")
                        continue
                    else:
                        print(f"✗ Warning: Empty response after {max_retries} retries")
                        break
                
                # If we got a result, break the retry loop
                break
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"  Error on attempt {attempt + 1}: {e}")
                    print(f"  Retrying...")
                    continue
                else:
                    print(f"✗ Error processing {image_path.name} after {max_retries} retries: {e}")
                    result = None
                    break
        
        # If we have no result after retries, skip to next image
        if not result or not result.strip():
            continue
        
        # Parse the JSON response
        try:
            response_obj = json.loads(result)
            
            # Expect response to have a "questions" array
            if "questions" not in response_obj:
                print(f"✗ Warning: Response does not contain 'questions' array")
                print(f"Response preview: {result[:300]}...")
                continue
            
            updated_questions = response_obj["questions"]
            
            # Validate that we got the same number of questions
            if len(updated_questions) != len(questions):
                print(f"✗ Warning: Response has {len(updated_questions)} questions, expected {len(questions)}")
            
            # Update the questions list
            questions = updated_questions
            
            # Count filled answer_keys
            filled_count = sum(1 for q in questions if q.get("answer_key") is not None)
            print(f"✓ Answer keys filled: {filled_count}/{len(questions)}")
            
        except json.JSONDecodeError as e:
            print(f"✗ Warning: Response is not valid JSON: {e}")
            print(f"Response preview: {result[:300]}...")
            continue
    
    # Save updated questions back to the output file
    print(f"\nSaving updated questions to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")
    
    print(f"✓ Successfully saved {len(questions)} question(s)")
    
    # Calculate totals
    script_elapsed_time = time.time() - script_start_time
    total_cost = sum(call['cost'] for call in api_calls)
    
    # Build stage summary (single stage for fill_keys)
    stage_summary = {}
    if api_calls:
        stage_summary["FillKeys"] = {
            "time": sum(call['time'] for call in api_calls),
            "model": API_CONFIG["model"],
            "cost": total_cost
        }
    
    # Print full cost report (detailed table of all API calls)
    print_full_cost_report(api_calls, total_cost)
    
    # Print summary cost report (compact table by stage) - FINAL report before prompt
    print_summary_cost_report(stage_summary, script_elapsed_time, total_cost)
    
    # Ask if user wants to clear the /keys directory
    print("\n")
    clear_response = input("Do you want to clear the /keys directory? (y/n): ").strip().lower()
    
    if clear_response == 'y':
        try:
            # Delete all files in the keys directory
            deleted_count = 0
            for file_path in KEYS_DIR.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            print(f"✓ Cleared {deleted_count} file(s) from {KEYS_DIR}")
        except Exception as e:
            print(f"✗ Error clearing directory: {e}")
    else:
        print("Keys directory was not cleared.")


if __name__ == "__main__":
    main()
