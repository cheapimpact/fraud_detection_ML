import pandas as pd
import os

folder = 'olah'
files = os.listdir(folder)
for f in files:
    path = os.path.join(folder, f)
    if f.endswith('.csv'):
        df = pd.read_csv(path)
        print(f"{f}: {df.columns.tolist()}")
    elif f.endswith('.xlsx'):
        df = pd.read_excel(path)
        print(f"{f}: {df.columns.tolist()}")
