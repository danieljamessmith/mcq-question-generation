"""
Clear script - Empties output files and img/keys directories
"""
import os
from pathlib import Path


def clear_files_and_dirs():
    """Clear all output files and the img/keys directories."""
    
    # Get script directory
    root_dir = Path(__file__).parent
    
    # Define paths - all output files from each stage
    # Files use numeric prefixes (0_, 1_, 2_, 3_) for natural sort order
    files_to_clear = [
        # Stage 1: Transcription
        root_dir / "1-transcription" / "output.jsonl",
        # Stage 2: Generation (input, generator, corrector, validator outputs)
        root_dir / "2-generation" / "0_input.jsonl",
        root_dir / "2-generation" / "1_gen.json",
        root_dir / "2-generation" / "1_gen.jsonl",
        root_dir / "2-generation" / "2_corrected.json",
        root_dir / "2-generation" / "2_corrected.jsonl",
        root_dir / "2-generation" / "3_validated.json",
        root_dir / "2-generation" / "3_validated.jsonl",
        # Stage 3: Extraction
        root_dir / "3-extraction" / "output.tex",
        root_dir / "3-extraction" / "output_raw.tex",
        root_dir / "3-extraction" / "input_extract.jsonl",
    ]
    
    img_dir = root_dir / "1-transcription" / "img"
    keys_dir = root_dir / "1-transcription" / "keys"
    
    # Ask for confirmation
    print("This will empty the following files:")
    for file_path in files_to_clear:
        status = "✓ exists" if file_path.exists() else "✗ not found"
        print(f"  - {file_path.relative_to(root_dir)} ({status})")
    
    print(f"\nAnd clear all files in:")
    img_status = "✓ exists" if img_dir.exists() else "✗ not found"
    img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    print(f"  - {img_dir.relative_to(root_dir)} ({img_status}, {img_count} file(s))")
    
    keys_status = "✓ exists" if keys_dir.exists() else "✗ not found"
    keys_count = len(list(keys_dir.glob("*"))) if keys_dir.exists() else 0
    print(f"  - {keys_dir.relative_to(root_dir)} ({keys_status}, {keys_count} file(s))")
    
    # Get user confirmation
    response = input("\nAre you sure you want to proceed? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Operation cancelled.")
        return
    
    # Clear files
    print("\nClearing files...")
    for file_path in files_to_clear:
        try:
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Empty the file (or create it if it doesn't exist)
            with open(file_path, 'w', encoding='utf-8') as f:
                pass  # Write nothing to empty the file
            
            print(f"  ✓ Cleared: {file_path.relative_to(root_dir)}")
        except Exception as e:
            print(f"  ✗ Error clearing {file_path.relative_to(root_dir)}: {e}")
    
    # Clear img directory
    print("\nClearing img directory...")
    try:
        if img_dir.exists():
            deleted_count = 0
            for file_path in img_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            print(f"  ✓ Deleted {deleted_count} file(s) from {img_dir.relative_to(root_dir)}")
        else:
            print(f"  ⚠ Directory not found: {img_dir.relative_to(root_dir)}")
    except Exception as e:
        print(f"  ✗ Error clearing directory: {e}")
    
    # Clear keys directory
    print("\nClearing keys directory...")
    try:
        if keys_dir.exists():
            deleted_count = 0
            for file_path in keys_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
            print(f"  ✓ Deleted {deleted_count} file(s) from {keys_dir.relative_to(root_dir)}")
        else:
            print(f"  ⚠ Directory not found: {keys_dir.relative_to(root_dir)}")
    except Exception as e:
        print(f"  ✗ Error clearing directory: {e}")
    
    print("\n✓ Clear operation completed!")


def main():
    """Main entry point."""
    print("="*60)
    print("CLEAR SCRIPT - Empty output files and img/keys directories")
    print("="*60)
    print()
    
    try:
        clear_files_and_dirs()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")


if __name__ == "__main__":
    main()
