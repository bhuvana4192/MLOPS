import subprocess
import sys
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
pipeline_dir = Path(__file__).resolve().parent

print("Starting ML pipeline...\n")

for script_name in ["Preprocess.py", "train.py", "Predict.py"]:
    script_path = pipeline_dir / script_name
    subprocess.run([sys.executable, str(script_path)], cwd=str(project_dir), check=True)

print("\nPipeline Executed successfully.")