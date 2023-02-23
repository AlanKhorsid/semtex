from bs4 import BeautifulSoup
from lxml import etree
from Levenshtein import distance as lev
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36",
    "Accept-Language": "en-US, en;q=0.5",
}


def get_string_permutations(s):
    s = s.split()
    n = len(s)
    result = []
    for i in range(1, 2**n):
        subset = [s[j] for j in range(n) if (i & (1 << j))]
        if len(subset) > 1:
            result.append(" ".join(subset))
    if len(s) == 0 or len(result) == 0:
        result.append(None)
    result.extend(s)
    return result


def get_suggested(soup, query):
    try:
        awre_line = soup.find("a", href=lambda href: href and "AWRE" in href)["href"]
        query = awre_line.split("=")[1].split("&")[0]
        query = query.replace("+", " ")
        suggested_search = query
        print(suggested_search)
        return suggested_search
    except:
        return None


def get_results(dom):
    return dom.xpath('//*[@id="b_results"]/li/h2/a')


def find_best_match(results, query):
    BM_results_final = None
    min_leven = float("inf")
    for i in range(0, 3):
        print(f"{i}. {results[i].text}")
        perms = get_string_permutations(results[i].text.lower())
        for p in perms:
            leven_dist = lev(query.lower(), p.lower())
            if leven_dist == 0:
                # print(f"Found {p} in {query}")
                # print(f"Levenshtein distance: {leven_dist}")
                BM_results_final = p
                min_leven = 0
                break
            elif leven_dist < min_leven:
                min_leven = leven_dist
                BM_results_final = p

    print(f"Best match: {BM_results_final}")
    return BM_results_final


def get_query():
    query = input("Write a search query: ")
    return query


def get_final_result(query):
    url = f"https://www.bing.com/search?q={query}"
    webpage = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(webpage.content, "html.parser")
    dom = etree.HTML(str(soup))

    suggestion = get_suggested(soup, query)
    if not suggestion:
        print("No suggested search! Lets check the other results")
        results = dom.xpath('//*[@id="b_results"]/li/h2/a')
        find_best_match(results, query)
    return suggestion


while True:
    query = get_query()
    if not query:
        break
    get_final_result(query)
