"""Spinner utility for displaying loading animations during API calls."""

import time
import threading


class Spinner:
    """A context manager that displays a spinning animation while waiting."""
    
    # Unicode spinner frames (braille dots - works well across terminals)
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    # Fallback ASCII frames for terminals that don't support Unicode
    ASCII_FRAMES = ["|", "/", "-", "\\"]
    
    def __init__(self, message="Waiting for API response", use_unicode=True):
        self.message = message
        self.frames = self.FRAMES if use_unicode else self.ASCII_FRAMES
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _spin(self):
        """Background thread that animates the spinner."""
        idx = 0
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.frames[idx % len(self.frames)]
            # \r returns to start of line, end="" prevents newline
            print(f"\r  {frame} {self.message}... ({elapsed:.1f}s)", end="", flush=True)
            idx += 1
            time.sleep(0.1)
        # Clear the spinner line when done
        print("\r" + " " * 60 + "\r", end="", flush=True)
    
    def __enter__(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        return False

