import pandas as pd

file_name = "news_data_cleaned.csv"

try:
    # Load cleaned dataset
    df = pd.read_csv(file_name)

    # Total number of articles
    total_articles = len(df)

    # Unique news sources
    unique_sources = df["Source"].nunique()

    print("\nNews Dataset Analysis")
    print("-" * 30)
    print("Total News Articles:", total_articles)
    print("Unique News Sources:", unique_sources)

    # Optional: show source names
    print("\nSources List:")
    print(df["Source"].unique())

except FileNotFoundError:
    print("Error: news_data_cleaned.csv not found. Please run Day 8 first.")

except Exception as e:
    print("Error occurred:", e)
