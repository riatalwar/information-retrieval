import sys
import time
import requests
from recipe_scrapers import scrape_html
from typing import List
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import cloudscraper
from playwright.sync_api import sync_playwright


SEARCH_URL = "https://www.allrecipes.com/search?"
scraper = cloudscraper.create_scraper()


class Review:
    # TODO: figure out how to fetch the likes/tags (API call?)
    rating: int
    comment: str
    helpful: int
    
    def __init__(self, rating: int, comment: str, helpful: int):
        self.rating = rating
        self.comment = comment
        self.helpful = helpful

    def __str__(self):
        return str(self.rating) + " stars\t\t" + self.helpful + " found this review helpful\n" + self.comment + "\n"


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

    def __str__(self) -> str:
        output = self.title + " by " + self.author + "\n"
        for r in self.reviews:
            output += str(r)
        return output


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


def load_page_html(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        page.goto(url)

        # scroll so reviews are loaded
        for i in range(20):
            page.evaluate(f"window.scrollTo(0, {1500 * i})")
            page.wait_for_timeout(1)

        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    
        # get page with reviews loaded
        html = page.content()
        browser.close()
    return html


def parse_reviews(text: str, ct: int) -> List[Review]:
    soup = BeautifulSoup(text, "html.parser")

    i = 0
    fetched_reviews = []

    reviews = soup.find_all(class_="mm-recipes-ugc-shared-item-card--review")
    for r in reviews:
        # skip the reviews that show up in the featured reviews section to avoid duplication
        if r.find_parent(class_="mm-recipes-ugc-threaded-add-feedback__most-helpful"):
            continue

        body = r.find(class_="mm-recipes-ugc-shared-item-card__text").text.strip()
        helpful_ct = r.find(class_="mm-recipes-ugc-shared-helpful-button").text.strip()
        stars = r.find(class_="mm-recipes-ugc-shared-star-rating")
        rating = len([
            use for use in stars.find_all("use")
            # only get filled stars, ignore the empty ones
            if use.get("xlink:href") == "#ugc-shared-icon-star"
        ]) if stars else None

        review = Review(rating, body, helpful_ct)
        fetched_reviews.append(review)
        i += 1
        if i >= ct: return fetched_reviews
    return fetched_reviews


def scrape_page(url: str) -> Recipe:
    # fetch page html
    html = load_page_html(url)
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


def read_queries_doc(file) -> List[str]:
    queries = []
    with open(file) as f:
        for line in f:
            queries.append(line)
    return queries


def main():
    profile = sys.argv[1]
    query_file = sys.argv[2]
    queries = read_queries_doc(query_file)
    for q in queries:
        print("Processing query")
        recipes = find_recipes(q, 3) # find the first 3 recipes 
        print("Recipes found: ")
        for r in recipes:
            print(r)


if __name__ == '__main__':
    main()
