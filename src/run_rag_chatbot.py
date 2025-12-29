"""
Run RAG Chatbot Streamlit app
"""
import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Change to project root so paths work correctly
    os.chdir(project_root)
    
    # Run streamlit app with src in Python path
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/rag/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ])