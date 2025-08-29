import sys
from pathlib import Path

# Allow running the script directly without installing the package
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatbot.preprocess_cli import main

if __name__ == "__main__":
    main()
