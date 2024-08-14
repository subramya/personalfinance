import requests

url = "https://google-news13.p.rapidapi.com/search"

querystring = {"keyword":"facebook","lr":"en-US"}

headers = {
	"x-rapidapi-key": "0018fd2bb7msh239e9f3016c6c47p1af8d3jsn1196b6ea3c51",
	"x-rapidapi-host": "google-news13.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())