import requests

url = "https://www.wikidata.org/w/api.php"

while True:
    query = input("Enter name: ")
    if query == "quit":
        break
    else:
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": query,
            "limit": "10",
        }
        try:
            id_list = []
            data = requests.get(url, params=params)
            for i in range(len(data.json()["search"])):
                print(data.json()["search"][i]["label"])
                print(data.json()["search"][i]["id"])
                id_list.append(data.json()["search"][i]["id"])

            for j in range(len(id_list)):
                params = {
                    "action": "wbgetentities",
                    "languages": "en",
                    "format": "json",
                    "ids": id_list[j],
                    "props": "claims",
                }
                data = requests.get(url, params=params)
                print(id_list[j])
                print(data.json())
        except:
            print("Invalid Input try again !!!")
