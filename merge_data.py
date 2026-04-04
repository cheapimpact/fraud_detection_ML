import pandas as pd
import os

def main():
    folder = 'olah'
    
    print("Loading data...")
    # 1. Load Compiled Analysis Results
    f1_path = os.path.join(folder, 'Compiled_Analysis_Results.xlsx')
    df1 = pd.read_excel(f1_path)
    df1.rename(columns={'ticker': 'Ticker', 'year': 'Year'}, inplace=True)
    
    # 2. Load Dataset Volatilitas
    f2_path = os.path.join(folder, 'Dataset_Volatilitas.csv')
    df2 = pd.read_csv(f2_path)
    # Clean Ticker and extract Year
    df2['Ticker'] = df2['Ticker'].astype(str).str.replace('.JK', '', regex=False)
    df2['Year'] = pd.to_datetime(df2['Event_Target_Date'], errors='coerce').dt.year
    
    # 3. Load Raw Data Dump
    f3_path = os.path.join(folder, 'Raw_Data_Dump_2022-2024 No Outlier.xlsx')
    df3 = pd.read_excel(f3_path)
    
    print("Standardizing join keys...")
    # Standardize Ticker and Year for stable join
    for idx, df in enumerate([df1, df2, df3], start=1):
        df['Ticker'] = df['Ticker'].astype(str).str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        print(f"Dataframe {idx} shape: {df.shape}")
        
    print("Performing full outer join...")
    # Full out join df1 and df2
    merged = pd.merge(df1, df2, on=['Ticker', 'Year'], how='outer')
    # Full outer join with df3
    merged = pd.merge(merged, df3, on=['Ticker', 'Year'], how='outer')
    
    output_filename = 'Combined_Full_Outer_Join.xlsx'
    output_path = os.path.join(folder, output_filename)
    print(f"Saving merged data to {output_path}...")
    merged.to_excel(output_path, index=False)
    print(f"Merge successful! Final shape: {merged.shape}")

if __name__ == '__main__':
    main()
