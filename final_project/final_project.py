import sys
import time
import requests
from recipe_scrapers import scrape_html
from typing import List
from urllib.parse import urlencode
from urllib.request import urlopen
from bs4 import BeautifulSoup
import cloudscraper


SEARCH_URL = "https://www.allrecipes.com/search?"


class Recipe:
    title: str
    author: str
    category: str
    time: float
    yields: int
    ingredients: List[str]
    instructions: str
    ratings: List
    cuisine: str
    desc: str
    url: str

    def repr(self) -> str:
        return (f"title: {self.title}\n" +
            f"  author: {self.author}\n" +
            f"  {self.desc}\n" +
            f"  link: {self.url}")


def read_query(query: str):
    words = query.strip().split(' ')
    # do any preprocessing, get rid of filler words?
    return words


def read_profile(profile: str):
    # get user profile
    pass


def rank_recipe():
    # given a recipe from allrecipes, rank with similarity to query and profile
    pass


def scrape_page(url: str) -> Recipe:
    html = urlopen(url).read().decode("utf-8") 
    scraper = scrape_html(html, url)
    return Recipe(
        scraper.title(),
        scraper.author(),
        scraper.category(),
        scraper.total_time(),
        scraper.yields(),
        scraper.ingredients(),
        scraper.instructions(),
        scraper.ratings(),
        scraper.cuisine(),
        scraper.description(),
        url
    )


def find_recipes(query: str, ct: int):
    # search for a recipe on allrecipes
    params = { "q": query }

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.google.com/",  
        "DNT": "1",
    }
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get("https://www.allrecipes.com")

    time.sleep(2)

    scraper = cloudscraper.create_scraper()
    response = scraper.get(SEARCH_URL + urlencode(params))

    # fetch urls corresponding to recipes
    soup = BeautifulSoup(response.text, "html.parser")
    i = 0
    links = []

    for link in soup.select("a[href*='/recipe/']"):
        href = link.get("href")
        if not href or href in links:
            continue
        links.append(href)
        i += 1
        if i >= ct:
            break

    return links


def read_queries_doc(file):
    with open(file) as f:
        content = f.read()
        return content


def main():
    profile = sys.argv[1]
    query = sys.argv[2]
    words = read_queries_doc(query)
    links = find_recipes(words, 8)
    print(links)


if __name__ == '__main__':
    main()
