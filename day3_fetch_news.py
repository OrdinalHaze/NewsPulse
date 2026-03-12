import requests

api_key = "8f699a3ccdb149abb366f107cb93ba24"

url = f"https://newsapi.org/v2/everything?q=india&pageSize=5&apiKey={api_key}"


response = requests.get(url)
data = response.json()

# Print full response for debugging
print("API Response Status:", data.get("status"))
print("Total Results:", data.get("totalResults"))
print(data)   # This shows actual error if any

print("\nTop Headlines:\n")

if data.get("status") == "ok" and data.get("articles"):
    for i, article in enumerate(data["articles"], start=1):
        print(f"{i}. {article['title']}")
else:
    print("No news found or API error.")

