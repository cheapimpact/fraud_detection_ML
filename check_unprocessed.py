import os

base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bei_process")
unprocessed = []
processed = []

for item in sorted(os.listdir(base_dir)):
    item_path = os.path.join(base_dir, item)
    if os.path.isdir(item_path):
        files_in_dir = os.listdir(item_path)
        has_analysis_result = any(f.startswith("analysis_result") for f in files_in_dir)
        
        if has_analysis_result:
            processed.append(item)
        else:
            unprocessed.append(item)

print("\nUnprocessed folders:")
for u in unprocessed:
    print(" - " + u)

print(f"\nTotal folders: {len(processed) + len(unprocessed)}")
print(f"Processed: {len(processed)}")
print(f"Unprocessed: {len(unprocessed)}")
