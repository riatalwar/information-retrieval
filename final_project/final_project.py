import sys
import time
import requests
from recipe_scrapers import scrape_html
from typing import List
from urllib.parse import urlencode
from bs4 import BeautifulSoup
import cloudscraper
import numpy as np
from sentence_transformers import SentenceTransformer
from playwright.sync_api import sync_playwright


SEARCH_URL = "https://www.allrecipes.com/search?"
scraper = cloudscraper.create_scraper()
_encoder = SentenceTransformer("all-MiniLM-L6-v2")


class User:
    name: str
    dietary_restrictions: List[str]
    preferred_cuisines: List[str]
    disliked_ingredients: List[str]
    liked_ingredients: List[str]
    liked_recipes: List["Recipe"]  # list of recipe URLs

    def __init__(self, name: str, dietary_restrictions: List[str] = None,
                 preferred_cuisines: List[str] = None, disliked_ingredients: List[str] = None,
                 liked_ingredients: List[str] = None):
        self.name = name
        self.dietary_restrictions = dietary_restrictions or []
        self.preferred_cuisines = preferred_cuisines or []
        self.disliked_ingredients = disliked_ingredients or []
        self.liked_ingredients = liked_ingredients or []
        self.liked_recipes = []

    def add_liked_recipe(self, recipe: "Recipe"):
        if recipe not in self.liked_recipes:
            self.liked_recipes.append(recipe)


def build_user_profile() -> User:
    name = input("Enter your name: ").strip()

    print("Enter dietary restrictions one at a time (blank line when done):")
    dietary_restrictions = []
    while (val := input("  > ").strip()):
        dietary_restrictions.append(val)

    print("Enter preferred cuisines one at a time (blank line when done):")
    preferred_cuisines = []
    while (val := input("  > ").strip()):
        preferred_cuisines.append(val)

    print("Enter disliked ingredients one at a time (blank line when done):")
    disliked_ingredients = []
    while (val := input("  > ").strip()):
        disliked_ingredients.append(val)

    print("Enter liked ingredients one at a time (blank line when done):")
    liked_ingredients = []
    while (val := input("  > ").strip()):
        liked_ingredients.append(val)

    return User(name, dietary_restrictions, preferred_cuisines, disliked_ingredients, liked_ingredients)


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
        output += str(self.ingredients) + '\n'
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


def rank_recipes(query: str, recipes: List["Recipe"], user: User, count: int = 3) -> List["Recipe"]:
    def has_disliked(recipe: "Recipe") -> bool:
        ingredients_text = " ".join(recipe.ingredients).lower()
        return any(d.lower() in ingredients_text for d in user.disliked_ingredients)

    filtered = [r for r in recipes if not has_disliked(r)]
    if not filtered:
        return []

    def recipe_text(recipe: "Recipe") -> str:
        parts = [recipe.title or "", recipe.desc or "", " ".join(recipe.ingredients), recipe.cuisine or ""]
        return " ".join(p for p in parts if p)

    query_emb = _encoder.encode(query, convert_to_numpy=True)
    recipe_embs = _encoder.encode([recipe_text(r) for r in filtered], convert_to_numpy=True)

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    recipe_norms = recipe_embs / (np.linalg.norm(recipe_embs, axis=1, keepdims=True) + 1e-10)
    scores = (recipe_norms @ query_norm).tolist()

    INGREDIENT_BOOST = 0.05
    CUISINE_BOOST = 0.10

    for i, recipe in enumerate(filtered):
        ingredients_text = " ".join(recipe.ingredients).lower()
        for liked in user.liked_ingredients:
            if liked.lower() in ingredients_text:
                scores[i] += INGREDIENT_BOOST
        if recipe.cuisine:
            cuisine_lower = recipe.cuisine.lower()
            for pref in user.preferred_cuisines:
                if pref.lower() in cuisine_lower or cuisine_lower in pref.lower():
                    scores[i] += CUISINE_BOOST

    ranked = sorted(zip(scores, filtered), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked][:count]


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
    profile = build_user_profile()
    query_file = sys.argv[1]
    queries = read_queries_doc(query_file)
    for q in queries:
        print("Processing query:", q.strip())
        recipes = find_recipes(q, 10)
        ranked = rank_recipes(q, recipes, profile)
        print(f"Results ({len(ranked)} after filtering):")
        for r in ranked:
            print(r)


if __name__ == '__main__':
    main()
