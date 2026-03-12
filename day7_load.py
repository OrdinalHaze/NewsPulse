import pandas as pd

file_name = "news_data.csv"

try:
    # Load CSV file
    df = pd.read_csv(file_name)

    print("\nNews Data (First 5 Rows):\n")
    print(df.head())   # Shows first 5 rows

    print("\nData Information:\n")
    print("Total Articles:", len(df))
    print("Columns:", list(df.columns))

except FileNotFoundError:
    print("Error: news_data.csv not found. Please run Day 6 first.")
