# HW2 PT2 Writeup

## Stemmed vs Unstemmed
_See OUTPUT.md for the requested data_

When comparing the stemmed and unstemmed methods, I immediately noticed an improvement when stemming words. Not only were more relevant documents typically retrieved or higher ranked (not always, but often), but similarity scores across the board were higher. As for the reasoning behind this result, it makes logical sense that similarity would increase, as comparisons such as "jumped," "jumps," and "jumping" would all be stemmed to "jump" and increase the similarity score accordingly.

This shows how stemming can be particularly useful when we are looking for documents with similar topics. However, we may also consider the case where the endings serve an important function. For example, if we are looking for something strictly in the past, we may want more similarity to words ending in "ed," and not "ing" or "s." Though the overall content might be similar, endings do serve a function in language to differentiate between certain things, and their removal may result in some loss of precision.

## Retrieval Terms
When looking at the terms that influenced retrieval, I found that a much smaller number of terms (1-3 in most cases) than expected were typically shared between query and document to influence retrieval. After further consideration, I thought this was most likely the result of queries being fairly short, meaning that they simply don't have that many terms to begin with. I also was initially confused by the appearance of periods and commas before realizing that, since they are tokenized as individual entities and don't fall under the category of stopwords, they simply are just so common that they're bound to show up. However, the fact that they would be common in any document means that the actual keywords would have much more influence in deciding which documents to retrieve, and the punctuation should be more just background noise.

## Other Factors
_See ANALYSIS.md for additional analysis data_

After exploring some of the designated combinations, I ran some additional tests to compile more data on how the different fields impacted the accuracy of the retrieval. I used the same three queries as above (6, 9, 22) to compute the average number of relevant documents retrieved for each combination of factors.
One thing that I quickly noticed was that the cosine and overlap metrics performed better than dice and jaccard with reasonable consistency, often picking up relevant documents where the others failed.
In addition to this, tf-idf performed consistently better in retrieving documents compared to just tf or boolean vectors. This makes sense, as tf-idf is a much more involved, insightful way of computing the term vectors. Not only does it account for appearance and frequency within the desired document, it is also influenced by a term's overall frequency within the document corpus. This accounts for cases where a word might have high frequency in a document but is relatively meaningless since it has high frequency in every document; this is compared to words with high frequency within the document, but low frequency elsewhere, indicating higher significance.
Lastly, I found that removing stopwords didn't have as noticeable a benefit as I had expected. Take the following cases:

| Remove Stopwords | Avg. Relevant Docs |
|------------------|--------------------|
| FALSE	           | 4.6667             |
| FALSE	           | 4                  |
| FALSE	           | 3.6667             |
| TRUE	           | 4.3333             |
| TRUE	           | 4                  |
| TRUE	           | 3.6667             |

All other factors remain the same across these cases (tf-idf with stemming and cosine similarity), but the number of relevant documents retrieved is not noticeably different--sometimes not removing stopwords results in better recall, and other times it's the reverse. One potential explanation could be that tf-idf should account for this in some ways, as stopwords should be relatively consistent in frequency across documents and down-weighted accordingly.

## AI Usage
I used AI to help me understand the initial project setup and walk through the existing functions, meaning of various parameters, and intended outputs. I implemented the similarity functions, precision and recall computations, and term vectors without the use of AI in part 1. For part 2, I used AI to write the functions that gathered the data that I needed, as I could much more quickly look at the results with proper formatting. Using those results, I completed the analysis and writeup without the use of AI. In some cases, I used AI to write docstrings to help me better understand and organize my functions.