import requests

api_key = "8f699a3ccdb149abb366f107cb93ba24"

# Reliable endpoint
url = f"https://newsapi.org/v2/everything?q=technology&pageSize=5&apiKey={api_key}"

response = requests.get(url)
data = response.json()

print("\nNews Details:\n")

if data["articles"]:
    for i, article in enumerate(data["articles"], start=1):
        title = article["title"]
        source = article["source"]["name"]
        date = article["publishedAt"]

        print(f"News {i}")
        print("Title:", title)
        print("Source:", source)
        print("Published Date:", date)
        print("-" * 50)
else:
    print("No news found.")
