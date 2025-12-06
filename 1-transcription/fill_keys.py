import os
import json
import base64
import time
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
    
    # Call OpenAI Vision API with GPT-5
    response = client.chat.completions.create(
        model="gpt-5.1",
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
        reasoning_effort="medium",
        max_completion_tokens=16000,
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
        print(f"  Token usage - Input: {input_tokens:,}, Output: {output_tokens:,}, Time: {elapsed_time:.2f}s")
    
    print(f"✓ Response received successfully")
    
    return result, input_tokens, output_tokens


def main():
    """Main function to fill answer keys from answer key images."""
    # Start overall timing
    script_start_time = time.time()
    
    # Initialize token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    
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
                result, input_tokens, output_tokens = fill_answer_keys(
                    questions, image_path, prompt_text
                )
                
                # Track token usage
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
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

