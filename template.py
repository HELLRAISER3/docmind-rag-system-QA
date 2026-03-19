import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/components/chunk.py",
    f"src/components/parse.py",
    f"src/components/embed.py",
    f"src/components/retrieve.py",
    f"src/components/generate.py",
    f"src/utils/__init__.py",
    f"src/utils/common.py",
    f"src/pipeline/__init__.py",
    f"src/pipeline/chunk_01_pipeline.py",
    f"src/pipeline/parse_02_pipeline.py",
    f"src/pipeline/embed_03_pipeline.py",
    f"src/pipeline/retrieve_04_pipeline.py",
    f"src/pipeline/generate_05_pipeline.py",
    f"app/__init__.py",
    f"app/main.py",
    "Dockerfile",
    "requirements.txt",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    
    else:
        logging.info(f"{filename} is already exists")