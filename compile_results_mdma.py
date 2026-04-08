import os
import json
import pandas as pd

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bei_process")
output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Compiled_Analysis_Results.xlsx")

data = []

for item in sorted(os.listdir(base_dir)):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        json_path = os.path.join(item_path, "analysis_result.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    data.append({
                        "target_file": result.get("target_file"),
                        "ticker": result.get("ticker", item),
                        "mdma_start": result.get("mdma_start"),
                        "mdma_end": result.get("mdma_end"),
                        "Positive_Sum": result.get("Positive_Sum"),
                        "Negative_Sum": result.get("Negative_Sum"),
                        "Total_Matched_Words": result.get("Total_Matched_Words"),
                        "Total_Word": result.get("Total_Word"),
                        "processed_at": result.get("processed_at"),
                    })
            except Exception as e:
                print(f"Error reading {json_path}: {e}")

if data:
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Successfully compiled {len(data)} results into {output_file}")
else:
    print("No analysis_result.json files found.")
