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

The LLM shortens this query to "creamy cheesy vegetable pasta", so we searched for this in Allrecipes, and ranked the first 7 recipes based on our own judgement:

1. Stuffed Shells
2. Ground Beef Casserole
3. Butternut Squash Mac and Cheese
4. Easy Slow Cooker Tuna Casserole
5. Cheesy Sausage Pasta
6. Quick and Easy Chicken Spaghetti
7. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her


We started off with equal weights for reviews, ratings, liked/disliked ingredients, and preferred cusine:

1. Stuffed Shells -- Score: 5.24
2. Easy Slow Cooker Tuna Casserole -- Score: 4.84 (g)
3. Quick and Easy Chicken Spaghetti -- Score: 4.536
4. Ground Beef Casserole -- Score: 4.058 (g)
5. Butternut Squash Mac and Cheese -- Score: 3.43 (g)
6. Cheesy Sausage Pasta -- Score: 3.334 (g)
7. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her -- Score: 2.165 (g)

The rankings were somewhat similar to our own rankings, although it seemed that a few recipes with many recent negative reviews or a lower rating were ranking higher, despite the chicken spaghetti recipe fell to the bottom due to it's very low recent review ratings.
There was a cheese sauce recipe, but it was ranked lower in the list, which was expected as it is not as relevant.

We then adjusted the weights of each of the categories:  
REVIEW_SIM_WEIGHT = 1.0   
REVIEW_RATING_WEIGHT = 5.0  
RATING_SCALE = 20.0        
INGREDIENT_BOOST = 0.05  
INGREDIENT_PENALTY = 0.05  
CUISINE_BOOST = 0.10  

1. Easy Slow Cooker Tuna Casserole -- Score: 5.99
2. Butternut Squash Mac and Cheese -- Score: 5.91
3. Stuffed Shells -- Score: 5.448
4. Cheesy Sausage Pasta -- Score: 5.342
5. Ground Beef Casserole -- Score: 5.151
6. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her -- Score: 3.455 
7. Quick and Easy Chicken Spaghetti -- Score: 1.642  

We realized here that although recipes with lower ratings were being penalized, recipes with disliked ingredients (marked with a g) were being weighted too highly due to other factors such as reviews or ratings, so we then decided to increase the penalty of a disliked ingredient:  

REVIEW_SIM_WEIGHT = 1.0   
REVIEW_RATING_WEIGHT = 5.0  
RATING_SCALE = 20.0       
INGREDIENT_BOOST = 0.05  
INGREDIENT_PENALTY = 1.0  
CUISINE_BOOST = 0.10    

1. Stuffed Shells by Renee -- Score: 5.448
2. Easy Slow Cooker Tuna Casserole -- Score: 5.04
3. Butternut Squash Mac and Cheese -- Score: 4.96
4. Cheesy Sausage Pasta -- Score: 4.392
5. Ground Beef Casserole -- Score: 4.201
6. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her -- Score: 2.505 g
7. Quick and Easy Chicken Spaghetti -- Score: 1.642  

These results overall matched our own rankings the best, although the chicken spaghetti recipe was being penalized too harshly due to recent reviews, so we decreased the weight of review ratings:  

REVIEW_SIM_WEIGHT = 1.0   
REVIEW_RATING_WEIGHT = 3.0  
RATING_SCALE = 20.0      
INGREDIENT_BOOST = 1.0  
INGREDIENT_PENALTY = 1.0  
CUISINE_BOOST = 0.10

1. Stuffed Shells -- Score: 5.534
2. Easy Slow Cooker Tuna Casserole -- Score: 4.94
3. Ground Beef Casserole -- Score: 4.415
4. Butternut Squash Mac and Cheese -- Score: 3.91
5. Cheesy Sausage Pasta -- Score: 3.626
6. Quick and Easy Chicken Spaghetti -- Score: 3.279
7. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her -- Score: 2.335
