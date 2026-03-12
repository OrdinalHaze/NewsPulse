import requests
import pandas as pd

api_key = "8f699a3ccdb149abb366f107cb93ba24"

# Reliable endpoint
url = f"https://newsapi.org/v2/everything?q=technology&pageSize=10&apiKey={api_key}"

response = requests.get(url)
data = response.json()

# List to store news
news_list = []

if data.get("status") == "ok" and data.get("articles"):
    for article in data["articles"]:
        news_list.append({
            "Title": article["title"],
            "Description": article["description"],
            "Source": article["source"]["name"],
            "Date": article["publishedAt"]
        })

    # Convert to DataFrame
    df = pd.DataFrame(news_list)

    # Save to CSV
    df.to_csv("news_data.csv", index=False)

    print("News data saved successfully!")
    print("Total articles saved:", len(df))

else:
    print("No news data available.")
