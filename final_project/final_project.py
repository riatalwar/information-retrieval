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
    # might not need all of these
    title: str
    author: str
    category: str
    time: float
    ingredients: List[str]
    instructions: str
    rating: List
    cuisine: str
    desc: str
    url: str

    def __init__(self, title, author, category, time, ingredients, instructions, rating, cuisine, desc, url):
        self.title = title
        self.author = author
        self.category = category
        self.time = time
        self.ingredients = ingredients
        self.instructions = instructions
        self.rating = rating
        self.cuisine = cuisine
        self.desc = desc
        self.url = url


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
    # fetch page html
    scraper = cloudscraper.create_scraper()
    html = scraper.get(url).text

    recipe_scraper = scrape_html(html, url)
    return Recipe(
        recipe_scraper.title(),
        recipe_scraper.author(),
        recipe_scraper.category(),
        recipe_scraper.total_time(),
        recipe_scraper.ingredients(),
        recipe_scraper.instructions(),
        recipe_scraper.ratings(),
        recipe_scraper.cuisine(),
        recipe_scraper.description(),
        url
    )


def find_recipes(query: str, ct: int) -> List[Recipe]:
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
    recipes = []

    for link in soup.select("a[href*='/recipe/']"):
        href = link.get("href")
        if not href or href in links:
            continue
        try:
            recipe = scrape_page(href)
            recipes.append(recipe)
            links.append(href)
        except Exception:
            continue
        i += 1
        if i >= ct:
            break

    return recipes


def read_queries_doc(file):
    with open(file) as f:
        content = f.read()
        return content


def main():
    profile = sys.argv[1]
    query = sys.argv[2]
    words = read_queries_doc(query)
    recipes = find_recipes(words, 8)


if __name__ == '__main__':
    main()
