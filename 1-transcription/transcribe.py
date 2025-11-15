import os
import json
import base64
import time
import shutil
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
IMG_DIR = SCRIPT_DIR / "img"
OUTPUT_FILE = SCRIPT_DIR / "output.jsonl"
PROMPT_FILE = SCRIPT_DIR / "prompt_transcribe.txt"
TEMPLATE_FILE = SCRIPT_DIR.parent / "json template.txt"


def encode_image(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_text_file(file_path):
    """Load text content from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def transcribe_image(image_path, prompt_text, template_text, special_prompt="", retry_count=0):
    """Send image to OpenAI Vision API with prompt and template."""
    # Start timing
    start_time = time.time()
    
    # Encode the image
    base64_image = encode_image(image_path)
    
    # Inject special prompt into the prompt text
    # Replace {special_prompt} placeholder with actual special prompt or "None" if empty
    if "{special_prompt}" in prompt_text:
        special_prompt_text = special_prompt if special_prompt.strip() else "None (use default rules)"
        prompt_with_special = prompt_text.replace("{special_prompt}", special_prompt_text)
    else:
        # Fallback: append special prompt if placeholder not found
        special_prompt_text = f"\n\n**SPECIAL INSTRUCTIONS:** {special_prompt}" if special_prompt.strip() else ""
        prompt_with_special = prompt_text + special_prompt_text
    
    # Construct the full prompt
    full_prompt = f"{prompt_with_special}\n\nTemplate structure:\n{template_text}"
    
    # Call OpenAI Vision API with GPT-5
    response = client.chat.completions.create(
        model="gpt-5",
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
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=5000,  # Increased for complex transcriptions
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
        print(f"  Token usage - Input: {input_tokens}, Output: {output_tokens}, Time: {elapsed_time:.2f}s")
    
    print(f"✓ Response received successfully")
    
    return result, input_tokens, output_tokens


def main():
    """Main function to process all images."""
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
    
    # Load prompt and template
    print("Loading prompt and template...")
    prompt_text = load_text_file(PROMPT_FILE)
    template_text = load_text_file(TEMPLATE_FILE)
    
    # Get all PNG images (sorted alphabetically as Windows Explorer shows)
    png_files = sorted(IMG_DIR.glob("*.png"))
    
    if not png_files:
        print(f"No PNG files found in {IMG_DIR}")
        return
    
    print(f"Found {len(png_files)} PNG files to process")
    
    # Process each image
    for idx, image_path in enumerate(png_files, 1):
        print(f"\nProcessing [{idx}/{len(png_files)}]: {image_path.name}")
        
        # Retry logic for failed attempts
        max_retries = 2
        result = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"  Retry attempt {attempt}/{max_retries}...")
                    time.sleep(2)  # Brief delay before retry
                
                # Transcribe the image
                result, input_tokens, output_tokens = transcribe_image(image_path, prompt_text, template_text, special_prompt)
                
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
                        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                            f.write(f"{{\"error\": \"Empty response after retries\", \"image\": \"{image_path.name}\"}}\n")
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
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{{\"error\": \"Processing failed after retries\", \"image\": \"{image_path.name}\", \"message\": {json.dumps(str(e))}}}\n")
                    result = None
                    break
        
        # If we have no result after retries, skip to next image
        if not result or not result.strip():
            continue
        
        # Parse and validate JSON (response_format ensures it's already JSON)
        try:
            json_obj = json.loads(result)
            
            # Verify it has the required fields
            required_fields = ["prompt", "choices"]
            if not all(field in json_obj for field in required_fields):
                print(f"✗ Warning: Missing required fields in response")
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{{\"error\": \"Missing fields\", \"image\": \"{image_path.name}\", \"response\": {json.dumps(json_obj)}}}\n")
                continue
            
            # Append to output file
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            
            print(f"✓ Successfully transcribed and saved")
            
        except json.JSONDecodeError as e:
            print(f"✗ Warning: Response is not valid JSON: {e}")
            print(f"Response preview: {result[:300]}...")
            
            # Save raw response for debugging
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(f"{{\"error\": \"Invalid JSON\", \"image\": \"{image_path.name}\", \"raw_response\": {json.dumps(result[:500])}}}\n")
    
    # Calculate total elapsed time
    script_elapsed_time = time.time() - script_start_time
    
    print(f"\n✓ All done! Results saved to {OUTPUT_FILE}")
    
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
    
    # Ask if user wants to clear the /img directory
    print("\n")
    clear_response = input("Do you want to clear the /img directory? (y/n): ").strip().lower()
    
    if clear_response == 'y':
        try:
            # Delete all files in the img directory
            deleted_count = 0
            for file_path in IMG_DIR.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            print(f"✓ Cleared {deleted_count} file(s) from {IMG_DIR}")
        except Exception as e:
            print(f"✗ Error clearing directory: {e}")
    else:
        print("Image directory was not cleared.")


if __name__ == "__main__":
    main()
