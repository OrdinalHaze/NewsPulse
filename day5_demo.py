import requests

api_key = "8f699a3ccdb149abb366f107cb93ba24"

# Using reliable endpoint
url = f"https://newsapi.org/v2/everything?q=technology&pageSize=5&apiKey={api_key}"

try:
    response = requests.get(url)
    data = response.json()

    # Check API status
    if data.get("status") == "ok" and data.get("articles"):
        print("\nTop 5 News Headlines:\n")
        
        for i, article in enumerate(data["articles"], start=1):
            print(f"{i}. {article['title']}")
    
    else:
        print("No news available at the moment. Please try again later.")

except Exception as e:
    print("Error occurred:", e)
