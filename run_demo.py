#!/usr/bin/env python3
"""Script to run the interactive demo."""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        import networkx
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def main():
    """Main function to run the demo."""
    print("Graph Generative Models - Interactive Demo")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Please install required dependencies:")
        print("pip install streamlit plotly networkx")
        return
    
    # Check if demo file exists
    demo_path = Path(__file__).parent / "demo.py"
    if not demo_path.exists():
        print(f"Demo file not found: {demo_path}")
        return
    
    # Run Streamlit demo
    print("Starting Streamlit demo...")
    print("The demo will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(demo_path), "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")


if __name__ == "__main__":
    main()
