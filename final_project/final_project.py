import json
import sys
import time
import requests
from recipe_scrapers import scrape_html
from typing import List
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import cloudscraper


SEARCH_URL = "https://www.allrecipes.com/search?"
scraper = cloudscraper.create_scraper()


class Review:
    # TODO: figure out how to fetch the likes/tags (API call?)
    rating: int
    comment: str
    likes: int
    tags: List[str]
    
    def __init__(self, rating: int, comment: str):
        self.rating = rating
        self.comment = comment


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
    reviews: List[Review]

    def __init__(self, title, author, category, time, ingredients, instructions, rating, cuisine, desc, url, reviews):
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
        self.reviews = reviews


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


def parse_reviews(text: str, ct: int) -> List[Review]:
    soup = BeautifulSoup(text, "html.parser")
    i = 0
    fetched_reviews = []
    for script_tag in soup.find_all("script", id="allrecipes-schema_1-0"):
        try:
            data = json.loads(script_tag.string)[0]
            reviews = data.get("review", [])
            if reviews:
                for review in reviews:
                    r = Review(
                        review["reviewRating"]["ratingValue"], 
                        review["reviewBody"])
                    fetched_reviews.append(r)
                    i += 1
                    if i >= ct: return
        except Exception:
            continue
    return fetched_reviews


def scrape_page(url: str) -> Recipe:
    # fetch page html
    html = scraper.get(url).text
    reviews = parse_reviews(html, 5) # placeholder, find first 5 reviews

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
        url,
        reviews
    )


def find_recipes(query: str, ct: int) -> List[Recipe]:
    # search for a recipe on allrecipes
    params = { "q": query }

    # pretend to be a real user 
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
        if i >= ct: break

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
