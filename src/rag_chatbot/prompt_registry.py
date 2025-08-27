import os
from pathlib import Path

prompt_dir = "configs/prompts/"
registry = {}

for filename in os.listdir(prompt_dir):
    name_normalized = Path(filename).stem
    with open(os.path.join(prompt_dir, filename)) as f:
        registry[name_normalized] = f.read()
