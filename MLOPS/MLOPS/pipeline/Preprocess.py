from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
input_path = base_dir / "student-por.csv"
output_path = base_dir / "processed_student_data.csv"

df = pd.read_csv(input_path)
df.to_csv(output_path, index=False)

print(f"Data Preprocessing Completed. Saved to: {output_path}")