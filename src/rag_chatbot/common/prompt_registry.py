from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
registry = {}
for file in PROMPT_DIR.iterdir():
    if file.is_file():
        registry[file.stem] = file.read_text(encoding="utf-8")
