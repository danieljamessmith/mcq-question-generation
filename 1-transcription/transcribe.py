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
    "max_completion_tokens": 5000
}
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
    
    # Call OpenAI Vision API with spinner
    with Spinner("Transcribing image"):
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
                                "url": f"data:image/png;base64,{base64_image}"
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
    """Main function to process all images."""
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
    
    # Clear output file at the start
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    
    # Load prompt and template
    print("Loading prompt and template...")
    if not PROMPT_FILE.exists():
        print(f"Error: {PROMPT_FILE} not found")
        return
    if not TEMPLATE_FILE.exists():
        print(f"Error: {TEMPLATE_FILE} not found")
        return
    prompt_text = load_text_file(PROMPT_FILE)
    template_text = load_text_file(TEMPLATE_FILE)
    
    # Get all PNG images (sorted by creation date, oldest first)
    png_files = sorted(IMG_DIR.glob("*.png"), key=lambda x: x.stat().st_ctime)
    
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
                result, input_tokens, output_tokens, elapsed_time = transcribe_image(
                    image_path, prompt_text, template_text, special_prompt
                )
                
                # Track API call
                cost = calculate_cost(API_CONFIG["model"], input_tokens, output_tokens)
                api_calls.append({
                    "stage": "Transcribe",
                    "api_call": f"Transcribe-{idx}",
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
    
    # Calculate totals
    script_elapsed_time = time.time() - script_start_time
    total_cost = sum(call['cost'] for call in api_calls)
    
    print(f"\n✓ All done! Results saved to {OUTPUT_FILE}")
    
    # Build stage summary (single stage for transcribe)
    stage_summary = {}
    if api_calls:
        stage_summary["Transcribe"] = {
            "time": sum(call['time'] for call in api_calls),
            "model": API_CONFIG["model"],
            "cost": total_cost
        }
    
    # Print full cost report (detailed table of all API calls)
    print_full_cost_report(api_calls, total_cost)
    
    # Print summary cost report (compact table by stage) - FINAL report before prompt
    print_summary_cost_report(stage_summary, script_elapsed_time, total_cost)
    
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
