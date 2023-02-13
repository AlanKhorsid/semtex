import requests

url = "https://www.wikidata.org/w/api.php"


while True:
    query = input("Enter name : ")
    if query == "quit":
        break
    else:
        params = {
            "action": "wbgetentities",
            # "action": "wbsearchentities",
            "languages": "en",
            "format": "json",
            # "id": query
            "ids": "Q76"
        }
        try:
            data = requests.get(url, params=params)
            print(data.json())
            # print(data.json()["searchinfo"])
            # print(data.json()["search"][0]["description"])
            # print(data.json()["search"][0]["label"])
            # print(data.json()["search"][0]["id"])
            # print(data.json()["search"][0]["aliases"])

        except:
            print("Invalid Input try again !!!")
