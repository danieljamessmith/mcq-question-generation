"""
Clear script - Empties output files and img directory
"""
import os
from pathlib import Path


def clear_files_and_dirs():
    """Clear all output files and the img directory."""
    
    # Get script directory
    root_dir = Path(__file__).parent
    
    # Define paths
    files_to_clear = [
        root_dir / "1-transcription" / "output.jsonl",
        root_dir / "2-generation" / "gen.json",
        root_dir / "2-generation" / "gen.jsonl",
        root_dir / "2-generation" / "input.jsonl",
        root_dir / "3-extraction" / "output_raw.tex",
        root_dir / "3-extraction" / "input.jsonl",
        root_dir / "4-production" / "output.tex",
    ]
    
    img_dir = root_dir / "1-transcription" / "img"
    
    # Ask for confirmation
    print("This will empty the following files:")
    for file_path in files_to_clear:
        status = "✓ exists" if file_path.exists() else "✗ not found"
        print(f"  - {file_path.relative_to(root_dir)} ({status})")
    
    print(f"\nAnd clear all files in:")
    img_status = "✓ exists" if img_dir.exists() else "✗ not found"
    img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    print(f"  - {img_dir.relative_to(root_dir)} ({img_status}, {img_count} file(s))")
    
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
    
    print("\n✓ Clear operation completed!")


def main():
    """Main entry point."""
    print("="*60)
    print("CLEAR SCRIPT - Empty output files and img directory")
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

