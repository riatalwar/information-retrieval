# HW2 PT3 Writeup

_See OUTPUT_PT3.md for the requested data_

There were three main differences that I noticed in the results after applying singular value decomposition for latent semantic indexing.

## Higher Similarity Scores
Regardless of relevance, documents and queries had much higher similarity scores after performing SVD. This makes sense, as compressing the number of dimensions means that there are fewer components that need to be considered in similarity scores. When faced with the typical overwhelming number of terms, similarity can get quickly diluted by the many inconsequential terms. However, reducing the components to a small, critical set helps to clarify the similarities and differences that hold significant weight.

## Lower Precision and Recall
Unsurprisingly, compressing information results in some amount of loss of both precision and recall. Though many relevant documents were still retrieved, some the same as before SVD and some different, there was a small overall reduction in the counts of retrieved relevant documents. After graphing the singular values, I tried to choose a point that offered both reasonable levels of compression and accuracy, which requires some amount of compromise to achieve.

## Relevant Terms
As expected, the relevant "terms" are no longer exactly as you would expect. Since SVD effectively transforms documents and queries from being represented by term vectors to compressed component vectors, the "terms" lose some of their meeting as they encompass larger concepts that are not easily demonstrated.