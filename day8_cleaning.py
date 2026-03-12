import pandas as pd
import re

# Step 1: Load the CSV file
file_name = "news_data.csv"
df = pd.read_csv(file_name)

print("Original Data:")
print(df.head())
print("Total rows before cleaning:", len(df))


# Step 2: Remove empty rows (missing Title or Source)
df = df.dropna(subset=["Title", "Source"])


# Step 3: Remove duplicate news (based on Title)
df = df.drop_duplicates(subset="Title")


# Step 4: Clean unwanted symbols from text
def clean_text(text):
    text = str(text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["Title"] = df["Title"].apply(clean_text)

# If you have Description column, clean it too
if "Description" in df.columns:
    df["Description"] = df["Description"].apply(clean_text)


# Step 5: Save cleaned data
clean_file = "news_data_cleaned.csv"
df.to_csv(clean_file, index=False)

print("\nCleaning Completed!")
print("Total rows after cleaning:", len(df))
print("Cleaned file saved as:", clean_file)
