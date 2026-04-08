import pandas as pd
import os

def main():
    folder = 'olah'
    
    print("Loading data...")
    # 1. Load Compiled Analysis Results
    f1_path = os.path.join(folder, 'Compiled_Analysis_Results.xlsx')
    df1 = pd.read_excel(f1_path)
    df1.rename(columns={'ticker': 'Ticker', 'year': 'Year'}, inplace=True)
    df1 = df1[['Ticker', 'Year', 'Positive_Sum', 'Negative_Sum', 'Total_Matched_Words', 'Total_Word']]
    
    # 2. Load Dataset Volatilitas
    f2_path = os.path.join(folder, 'Data_Volatilitas_Horizontal.xlsx')
    df2 = pd.read_excel(f2_path)
    df2.rename(columns={'ticker': 'Ticker', 'year': 'Year', 'report date': 'Report Date'}, inplace=True)
    df2['Ticker'] = df2['Ticker'].astype(str).str.replace('.JK', '', regex=False)
    df2 = df2[['Ticker', 'Year', 'Report Date', 'Volatilitas 30 Hari SEBELUM', 'Volatilitas 30 Hari Setelah']]
    
    # 3. Load Raw Data Dump
    f3_path = os.path.join(folder, 'Raw_Data_Dump_2022-2024 No Outlier.xlsx')
    df3 = pd.read_excel(f3_path)
    cols3 = ['Ticker', 'Year', 'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_Score', 'Prediction']
    df3 = df3[cols3]
    
    print("Standardizing join keys...")
    # Standardize Ticker and Year for stable join
    for idx, df in enumerate([df1, df2, df3], start=1):
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        print(f"Dataframe {idx} shape: {df.shape}")
        
    print("Performing full outer join...")
    # Start with df3, then merge df2, then df1
    merged = pd.merge(df3, df2, on=['Ticker', 'Year'], how='outer')
    merged = pd.merge(merged, df1, on=['Ticker', 'Year'], how='outer')
    
    # Reorder columns as requested
    final_cols = [
        'Ticker', 'Year', 'Report Date', 
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_Score', 'Prediction', 
        'Volatilitas 30 Hari SEBELUM', 'Volatilitas 30 Hari Setelah', 
        'Positive_Sum', 'Negative_Sum', 'Total_Matched_Words', 'Total_Word'
    ]
    
    existing_cols = [c for c in final_cols if c in merged.columns]
    merged = merged[existing_cols]
    
    output_filename = 'Combined_Full_Outer_Join.xlsx'
    output_path = os.path.join(folder, output_filename)
    print(f"Saving merged data to {output_path}...")
    merged.to_excel(output_path, index=False)
    print(f"Merge successful! Final shape: {merged.shape}")

if __name__ == '__main__':
    main()
