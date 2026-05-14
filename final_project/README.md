### Setup Instructions:
1. export a new environment variable, `GEMINI_API_KEY`, with your Gemini API key
2. run `bash setup.sh` to download necessary dependencies
3. run the program with `python3 final_project.py <query_file>`

### Project Description
The goal of this project is to provide highly customized recipe recommendations based off of a detailed user profile and recipe description. The motivation comes from the fact that many recipe sites do not provide much filtering functionality, which can make it frustrating to look for a recipe that fits your tastes and/or dietary restrictions.     

This program takes in a user profile, consisting of ingredients the user does/does not like, dietary restrictions, preferred cuisines, and other preferences, as well as a text file consisting of recipe descriptions that the user would like to search for. The recipe descriptions are then simplified, then searched for in Allrecipes. The program then fetches the top results returned from Allrecipes, and compares them against the original query and user profile, taking into account reviews and ratings, to return re-ranked recommended recipes.    

We started with simply looking at a recipe's description, ingredients, and other information that could be found using an Allrecipes scraper that we found online, but later also decided to include descriptions, ratings, and helpful counts of reviews, as they often include valuable information about a recipe that are not always explicitly stated in the description, for example, recipe modifications that could include ingredients the user likes.

In regards to using pure recipe description and ingredients vs. the preferences of other users, we decided to go for a more recipe description-based approach, as comparing a user's profile to others in order to make recommendations would require an extensive fake user base that would take time to set up. We were also leaning more towards using the recipe descriptions due to the fact that it is difficult to place people into set categories, as it is possible people will have preferences that could be in the middle of two groups, since every person is unique. However, we did incorporate the opinons of other users in the form of reviews and ratings, where we look at the most recent reviews that other users have left and their ratings as well as the overall rating of the recipe. This approach would allow for us to take advantage of other people's feedback without relying completely on similarity between users. Feedback from others is valuable, after all, we are more likely to try a recipe if it is highly rated by others. It is also important because using purely recipe-based comparisons will sometimes miss things that the user may not have explicitly stated. For example, there could be a recipe that matches what the user wants very well based on description, but may taste odd due to the addition of one extra ingredient, which user may not have explicitly excluded that ingredient.

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

(Recipes containing garlic are labelled with a g)

The rankings were somewhat similar to our own rankings, although it seemed that a few recipes with many recent negative reviews or a lower rating were ranking higher, despite the chicken spaghetti recipe fell to the bottom due to it's very low recent review ratings. Stuffed Shells was at the top, which was desired due to it having no garlic and both of the liked ingredients.
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


We also tested the same query with a slightly modified user profile:
dietary restrictions: vegetarian
preferred cuisines: none
disliked ingredients: garlic 
liked ingredients: cheese, mushrooms

This was done in order to test whether dietary restrictions were being filtered, which, as we can see in the below results, was indeed done:

1. Stuffed Shells -- Score: 5.534  
2. Butternut Squash Mac and Cheese -- Score: 3.91  
3. My Mother-in-Law's Cheese Sauce...ssshh Don't Tell Her -- Score: 2.335  

All recipes with meat in the ingredients were filtered out.  

In order to test the effect of cuisine in rankings, we created another user profile:
dietary restrictions: none
preferred cuisines: Asian
disliked ingredients: none 
liked ingredients: none

And tested the query "a spicy rice dish with vegetables", which simplified down to "spicy vegetable rice".
Following the same process as above, we ranked the first 7 recipes that appear when searching this on Allrecipes:

1. Spicy Eggplant
2. Korean Spicy Chicken and Potato (Tak Toritang)
3. Sweet, Sticky, and Spicy Chicken
4. Spicy Crispy Beef
5. Orange Beef-Style Tofu Stir-Fry
6. Spicy African Yam Soup
7. Spicy Red Bell Pepper Soup

Initially, we decided that cuisine should not have too large of a boost on a recipe's ranking, as you can usually infer the cuisine of a recipe from its description, so we thought cuisine might be slightly redundant. 
We first ran the code with the following weights:

REVIEW_SIM_WEIGHT = 1.0   
REVIEW_RATING_WEIGHT = 3.0  
RATING_SCALE = 20.0      
INGREDIENT_BOOST = 1.0  
INGREDIENT_PENALTY = 1.0  
CUISINE_BOOST = 0.10

1. Spicy Eggplant -- Score: 3.99
2. Korean Spicy Chicken and Potato (Tak Toritang) -- Score: 3.838
3. Spicy Red Bell Pepper Soup -- Score: 3.729
4. Sweet, Sticky, and Spicy Chicken -- Score: 3.657
5. Spicy Crispy Beef -- Score: 3.341
6. Spicy African Yam Soup -- Score: 3.106
7. Orange Beef-Style Tofu Stir-Fry -- Score: 2.876

We decided after this that cuisine was not being weighted heavily enough, as there were a few recipes ranked rather high (Spicy Red Bell Pepper Soup) despite having no affiliation to Asian cuisine. Because of this, we decided to slightly increase the weight of the cuisine field:

REVIEW_SIM_WEIGHT = 1.0   
REVIEW_RATING_WEIGHT = 3.0  
RATING_SCALE = 20.0      
INGREDIENT_BOOST = 1.0  
INGREDIENT_PENALTY = 1.0  
CUISINE_BOOST = 0.50  

1. Spicy Eggplant -- Score: 4.39
2. Korean Spicy Chicken and Potato (Tak Toritang) -- Score: 3.838
3. Spicy Crispy Beef -- Score: 3.762
4. Spicy Red Bell Pepper Soup -- Score: 3.729
5. Sweet, Sticky, and Spicy Chicken -- Score: 3.657
6. Orange Beef-Style Tofu Stir-Fry -- Score: 3.276
7. Spicy African Yam Soup -- Score: 3.106
