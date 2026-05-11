### Setup Instructions:
1. export a new environment variable, `GEMINI_API_KEY`, with your Gemini API key
2. run `bash setup.sh` to download necessary dependencies
3. run the program with `python3 final_project.py <query_file>`

### Project Description
The goal of this project is to provide highly customized recipe recommendations based off of a detailed user profile and recipe description. The motivation comes from the fact that many recipe sites do not provide much filtering functionality, which can make it frustrating to look for a recipe that fits your tastes and/or dietary restrictions.     

This program takes in a user profile, consisting of ingredients the user does/does not like, dietary restrictions, preferred cuisines, and other preferences, as well as a text file consisting of recipe descriptions that the user would like to search for. The recipe descriptions are then simplified, then searched for in Allrecipes. The program then fetches the top results returned from Allrecipes, and compares them against the original query and user profile, taking into account reviews and ratings, to return re-ranked recommended recipes.    

We started with simply looking at a recipe's description, ingredients, and other information that could be found using an Allrecipes scraper that we found online, but later also decided to include descriptions, ratings, and helpful counts of reviews, as they often include valuable information about a recipe that are not always explicitly stated in the description, for example, recipe modifications that could include ingredients the user likes.

To evaulate effectiveness, we created a query, "pasta dish without tomato sauce, includes many vegetables and creamy, cheese-based sauce" and tested it with the following user profile:

dietary restrictions: peanuts, tree nuts
preferred cuisines: none
disliked ingredients: garlic 
liked ingredients: cheese, mushrooms

The LLM shortens this query to "creamy cheesy vegetable pasta", so we searched for this in Allrecipes, and ranked the first 10 recipes based on our own judgement:

1. Stuffed Shells
2. Stick of Butter Mississippi Chicken Spaghetti
3. Quick and Easy Chicken Spaghetti
4. Cheesy Polenta

Excluded due to presence of garlic:
1. Butternut Squash Mac and Cheese 
2. Italian Wedding Pasta Bake 
3. Ground Beef Casserole 
4. Cheesy Sausage Pasta 
5. Cheesy Kielbasa Pasta 
6. Easy Slow Cooker Tuna Casserole 

We started by weighing reviews, rating, ingredients, and cuisine equally, but we realized that this was weighing the reviews too heavily, which was a problem, especially since some of the longer and more detailed reviews were ones with lower ratings.
The returned results:
1. Quick and Easy Chicken Spaghetti
2. Stuffed Shells
3. Cheesy Polenta
4. Stick of Butter Mississippi Chicken Spaghetti

We then shifted the weight of the reviews to 2, the weight of ratings to 20.0, the weight of ingredients in the liked ingredients list to 0.05, and the weight of same cuisine stayed at 1, as we discovered that many recipes did not have a cuisine. The reviews were also individually weighted by the number of helpful votes they had, which we thought would be useful since a review with a higher helpful count means that more people agreed with its content. Furthermore, the reviews' ratings were taken into account, as a recipe with many recent low ratings is often less desirable. The one recipe that was in a position we did not anticipate was the "Stick of Butter Mississippi Chicken Spaghetti", which most likely was due to the fact that its reviews were very short and concise (eg "Delicious!"), resulting in a lower similarity with the query itself.

The new returned results:
1. Quick and Easy Chicken Spaghetti
2. Cheesy Polenta
3. Stuffed Shells
4. Stick of Butter Mississippi Chicken Spaghetti

We discovered here that while we did add review rating into the weight, it was still being overshadowed by similarity between review content and the query, so we decided to separate the weight of the review similarity and review rating so that the similarity was weighted 2 and the review rating was weighted 5:

1. Stuffed Shells
2. Quick and Easy Chicken Spaghetti
3. Stick of Butter Mississippi Chicken Spaghetti
4. Cheesy Polenta

The above results were much closer to our initial ranking.