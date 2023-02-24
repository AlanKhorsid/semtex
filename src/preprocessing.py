from bs4 import BeautifulSoup
from lxml import etree
from Levenshtein import distance as lev
import requests
from preprocesschecker import check_spellchecker_threaded

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36",
    "Accept-Language": "en-US, en;q=0.5",
}


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


def get_final_result(query):
    url = f"https://www.bing.com/search?q={query}"
    webpage = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(webpage.content, "html.parser")

    suggestion = get_suggested(soup, query)
    if not suggestion:
        return query
    return suggestion


# check_spellchecker(get_final_result)
check_spellchecker_threaded(get_final_result, num_threads=100)
