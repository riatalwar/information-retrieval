import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# Part 3 imports
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.feature_extraction import DictVectorizer

# autograder fix
import nltk
nltk.download('punkt_tab')


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    '''
    Computes the tf-idf vector for a document.

    Uses log-normalized term frequency: (1 + log10(tf)) * log10(N / df),
    N : total number of documents
    df : document frequency of each term

    Args:
        doc: The document to score.
        doc_freqs: A mapping from term to document frequency across the corpus.
        weights: Field weights passed through to compute_tf.

    Returns:
        A dict mapping each term to its tf-idf score.
    '''
    tf_computed = compute_tf(doc, doc_freqs, weights)   # get document term frequencies
    tfidf_computed = dict()

    # compute tf-idf for each term
    for word, tf in tf_computed.items():
        if tf == 0 or doc_freqs.get(word, 0) == 0: continue
        tfidf_computed[word] = (1 + np.log10(tf)) * np.log10(len(doc_freqs) / doc_freqs[word])

    return tfidf_computed

def compute_boolean(doc: Document, doc_freqs: Dict[str, int], weights: list):
    '''
    Computes a sparse boolean term vector for a document.

    Args:
        doc: The document to score.
        doc_freqs: A mapping from term to document frequency across the corpus.
        weights: Field weights passed through to compute_tf.

    Returns:
        A dict mapping each present term to 1.
    '''
    tf_computed = compute_tf(doc, doc_freqs, weights)
    bool_computed = dict()

    # store all terms with nonzero frequencies
    for word, tf in tf_computed.items():
        if tf == 0: continue    # don't bother storing term frequencies of 0
        bool_computed[word] = 1

    return bool_computed



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))

def dice_sim(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dice similarity between two sparse term vectors represented as dictionaries.
    '''
    num = 2 * dictdot(x, y)
    if num == 0:
        return 0
    denom = sum(x.values()) + sum(y.values())
    return num / denom

def jaccard_sim(x, y):
    '''
    Computes the jaccard similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    denom = sum(x.values()) + sum(y.values()) - num

    if num == 0 or denom == 0:
        return 0

    return num / denom

def overlap_sim(x, y):
    '''
    Computes the overlap similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    denom = min(sum(x.values()), sum(y.values()))
    return num / denom


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is an ordered list of document ids sorted by rank
    `relevant` is a list of relevant documents
    '''
    relevant_set = set(relevant)
    points = [(0, 1)]  # implicit 100% precision if no docs are returned
    if recall == 0: return 1
    num_relevant_found = 0

    # go through the document ranking to tally the relevant doc count at each level of returned docs
    for k, doc_id in enumerate(results, start=1):
        if doc_id in relevant_set:
            num_relevant_found += 1
            r = num_relevant_found / len(relevant)
            p = num_relevant_found / k
            points.append((r, p))

    points.sort() # sort points on recall level for search

    # locate either
    # a. pts on either side of desired recall level
    # b. exact recall
    i = 1
    prev_pt = points[0]
    while points[i][0] < recall:
        prev_pt = points[i]
        i += 1

    # return exact precision or fall back to interpolation
    if points[i][0] == recall: return points[i][1]

    return interpolate(prev_pt[0], prev_pt[1], points[i][0], points[i][1], recall)

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    sum = 0
    for i in range(10):
        sum += precision_at((i + 1) / 10, results, relevant)
    return sum / 10

def norm_recall(results, relevant):
    rel = len(relevant)
    n = len(results)

    # sum ranks of relevant documents
    sum_rank = 0
    for i in range(rel):
        sum_rank += results.index(relevant[i])

    sum_i = rel * (rel + 1) / 2
    denom = rel * (n - rel)
    
    return 1 - (sum_rank - sum_i) / denom

def norm_precision(results, relevant):
    rel = len(relevant)
    n = len(results)

    # normalized sum of ranks
    sum_rank = 0
    sum_i = 0
    for i in range(rel):
        sum_rank += np.log10(results.index(relevant[i]) + 1)
        sum_i += np.log10(i + 1)
        
    denom = n * np.log10(n) - (n - rel) * np.log10(n - rel) - rel * np.log10(rel) # approximate

    return 1 - (sum_rank - sum_i) / denom


### Extensions

# SVD
# to perform svd
# start with term-document matrix: term weights using tf, tf-idf, or boolean weighting for each document
# use scipy svd to compute the svd decomposition of the matrix--can adjust the k value to change dimensions
# the result will be a "doc-concept" and "term-concept" matrix from the decomposition
# each query will need to be transformed by the resulting decomposition to ensure dimensions are correct
# then, the various transformed vectors can be compared with our original similarity functions
# vectors will need to be transformed back to dict form to work with original functions
def svd(doc_vectors: List[Document], k: int):
    '''
    Peform SVD for latent semantic indexing, reducing to k dimensions
    
    Params
        doc_vectors: document-term mappings
        k: dimensions for resulting decomposition

    Returns
        vec: vector transformer
        U, sigma, Vt: matrix decomposition
    '''
    # convert doc vectors to a sparse term-document matrix
    vec = DictVectorizer(sparse=True)
    doc_term_mat = vec.fit_transform(doc_vectors)

    # perform svd to decompose matrix
    U, sigma, Vt = svds(doc_term_mat, k=k)

    return vec, U, sigma, Vt

def convert_query_vec(q_vec: Dict[str, float], vec: DictVectorizer, sigma, Vt):
    '''
    Convert standard query representation into compressed version

    Params
        vec: vector transformer
        sigma, Vt: matrix decomposition

    Returns
        Compressed query vector
    '''
    q_vec_tranf = vec.transform(q_vec)
    sig_inv = np.diag(1 / sigma)
    return q_vec_tranf @ Vt.T @ sig_inv

def plot_svd(sigma):
    '''
    Plot singular values from decomposition
    Allows examination of effectiveness as number of components increases

    Params
        sigma: kxk matrix from SVD
    '''
    plt.plot(sigma[::-1])
    plt.xlabel('Component')
    plt.ylabel('Singular value')
    plt.show()


### Search

def experiment(apply_svd=True):
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        # optionally run svd to transform document representation
        if apply_svd:
            # get decomposition
            vec, U, sigma, Vt = svd(doc_vectors, 300)
            # plot_svd(sigma) # -- decided on 300 components for strong predictions with reasonable compression
            # transform document vectors based on decomposition
            doc_vectors = vec.transform(doc_vectors) @ np.linalg.matrix_transpose(Vt) @ np.diag(sigma)
            doc_vectors = vec.inverse_transform(doc_vectors)    # back to sparse dict

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)

            # optionally use svd to transform query and generate results
            if apply_svd:
                query_vec = convert_query_vec(query_vec, vec, sigma, Vt)
                query_vec = vec.inverse_transform(query_vec)[0]
            
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        return  # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()

# AI-Generated helper functions to extract data
def query_top20_table(i: int, term='tfidf', sim='cosine', stem=True, removestop=True,
                      term_weights=TermWeights(author=1, title=3, keyword=4, abstract=1), apply_svd=True) -> str:
    '''
    Returns a markdown table of the top 20 retrieved documents for the i-th query.

    Args:
        i: 1-based query index.
        term: Term weighting scheme ('tf', 'tfidf', 'boolean').
        sim: Similarity function ('cosine', 'jaccard', 'dice', 'overlap').
        stem: Whether to stem tokens.
        removestop: Whether to remove stopwords.
        term_weights: Field weights for term scoring.
    '''
    term_funcs = {'tf': compute_tf, 'tfidf': compute_tfidf, 'boolean': compute_boolean}
    sim_funcs = {'cosine': cosine_sim, 'jaccard': jaccard_sim, 'dice': dice_sim, 'overlap': overlap_sim}

    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

    if apply_svd:
        # get decomposition
        vec, U, sigma, Vt = svd(doc_vectors, 300)
        # transform document vectors based on decomposition
        doc_vectors = vec.transform(doc_vectors) @ np.linalg.matrix_transpose(Vt) @ np.diag(sigma)
        doc_vectors = vec.inverse_transform(doc_vectors)    # back to sparse dict

    query = processed_queries[i - 1]
    query_vec = term_funcs[term](query, doc_freqs, term_weights)

    if apply_svd:
        query_vec = convert_query_vec(query_vec, vec, sigma, Vt)
        query_vec = query_vec = vec.inverse_transform(query_vec)[0]

    relevant = set(rels.get(query.doc_id, []))

    results_with_score = [(doc_id + 1, sim_funcs[sim](query_vec, doc_vec))
                          for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])[:20]

    lines = ['| Rank | Doc # | Title | Similarity |',
             '|------|-------|-------|------------|']
    for rank, (doc_id, score) in enumerate(results_with_score, start=1):
        title = ' '.join(docs[doc_id - 1].title) or '(no title)'
        if doc_id in relevant:
            title = f'**{title}**'
        lines.append(f'| {rank} | {doc_id} | {title} | {score:.4f} |')

    return '\n'.join(lines)


def query_top10_overlap(i: int, term='tfidf', sim='cosine', stem=True, removestop=True,
                        term_weights=TermWeights(author=1, title=3, keyword=4, abstract=1), apply_svd=True) -> str:
    '''
    Returns a markdown bullet list of overlapping terms (nonzero weight in both the query
    and document vectors) for each of the top 10 retrieved documents for the i-th query.

    Args:
        i: 1-based query index.
        term: Term weighting scheme ('tf', 'tfidf', 'boolean').
        sim: Similarity function ('cosine', 'jaccard', 'dice', 'overlap').
        stem: Whether to stem tokens.
        removestop: Whether to remove stopwords.
        term_weights: Field weights for term scoring.
    '''
    term_funcs = {'tf': compute_tf, 'tfidf': compute_tfidf, 'boolean': compute_boolean}
    sim_funcs = {'cosine': cosine_sim, 'jaccard': jaccard_sim, 'dice': dice_sim, 'overlap': overlap_sim}

    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    stopwords = read_stopwords('common_words')

    processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

    if apply_svd:
        # get decomposition
        vec, U, sigma, Vt = svd(doc_vectors, 300)
        # transform document vectors based on decomposition
        doc_vectors = vec.transform(doc_vectors) @ np.linalg.matrix_transpose(Vt) @ np.diag(sigma)
        doc_vectors = vec.inverse_transform(doc_vectors)    # back to sparse dict

    query = processed_queries[i - 1]
    query_vec = term_funcs[term](query, doc_freqs, term_weights)

    if apply_svd:
        query_vec = convert_query_vec(query_vec, vec, sigma, Vt)
        query_vec = query_vec = vec.inverse_transform(query_vec)[0]
        
    query_terms = set(query_vec.keys())

    results_with_score = [(doc_id + 1, sim_funcs[sim](query_vec, doc_vec), doc_vec)
                          for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])[:10]

    lines = []
    for rank, (doc_id, score, doc_vec) in enumerate(results_with_score, start=1):
        title = ' '.join(docs[doc_id - 1].title) or '(no title)'
        overlap = sorted(query_terms & set(doc_vec.keys()))
        lines.append(f'- **Rank {rank} — Doc {doc_id}** _{title}_')
        if overlap:
            for term in overlap:
                lines.append(f'  - {term} (query: {query_vec[term]:.4f}, doc: {doc_vec[term]:.4f})')
                if len(lines) > 10 * rank: break
        else:
            lines.append('  - (none)')

    return '\n'.join(lines)


def relevant_in_top20_table(query_ids: List[int]) -> str:
    '''
    For every configuration (term weighting x stem x removestop x similarity x term weights),
    counts the number of relevant documents retrieved in the top 20 results, averaged across
    the given queries. Returns a markdown table sorted by avg relevant descending.

    Args:
        query_ids: 1-based list of query indices to evaluate.
    '''
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {'tf': compute_tf, 'tfidf': compute_tfidf, 'boolean': compute_boolean}
    sim_funcs = {'cosine': cosine_sim, 'jaccard': jaccard_sim, 'dice': dice_sim, 'overlap': overlap_sim}

    permutations = [
        term_funcs,
        [False, True],   # stem
        [False, True],   # removestop
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
         TermWeights(author=1, title=3, keyword=4, abstract=1),
         TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    rows = []
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        counts = []
        for i in query_ids:
            query = processed_queries[i - 1]
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            relevant = set(rels[query.doc_id])
            counts.append(sum(1 for doc_id in results[:20] if doc_id in relevant))

        rows.append((term, stem, removestop, sim, ','.join(map(str, term_weights)), np.mean(counts)))

    rows.sort(key=lambda r: -r[5])

    lines = ['| Term | Stem | Remove Stopwords | Similarity | Term Weights | Avg Relevant in Top 20 |',
             '|------|------|------------------|------------|--------------|------------------------|']
    for term, stem, removestop, sim, weights, avg in rows:
        lines.append(f'| {term} | {stem} | {removestop} | {sim} | {weights} | {avg:.4f} |')

    return '\n'.join(lines)


def doc_top20_table(i: int, term='tfidf', sim='cosine', stem=True, removestop=True,
                    term_weights=TermWeights(author=1, title=3, keyword=4, abstract=1), apply_svd=True) -> str:
    '''
    Returns a markdown table of the top 20 documents most similar to document i
    (excluding itself).

    Args:
        i: 1-based document index.
        term: Term weighting scheme ('tf', 'tfidf', 'boolean').
        sim: Similarity function ('cosine', 'jaccard', 'dice', 'overlap').
        stem: Whether to stem tokens.
        removestop: Whether to remove stopwords.
        term_weights: Field weights for term scoring.
    '''
    term_funcs = {'tf': compute_tf, 'tfidf': compute_tfidf, 'boolean': compute_boolean}
    sim_funcs = {'cosine': cosine_sim, 'jaccard': jaccard_sim, 'dice': dice_sim, 'overlap': overlap_sim}

    docs = read_docs('cacm.raw')
    stopwords = read_stopwords('common_words')

    processed_docs, _ = process_docs_and_queries(docs, [], stem, removestop, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

    if apply_svd:
        # get decomposition
        vec, U, sigma, Vt = svd(doc_vectors, 300)
        # transform document vectors based on decomposition
        doc_vectors = vec.transform(doc_vectors) @ np.linalg.matrix_transpose(Vt) @ np.diag(sigma)
        doc_vectors = vec.inverse_transform(doc_vectors)    # back to sparse dict

    query_vec = doc_vectors[i - 1]
    results_with_score = [(doc_id + 1, sim_funcs[sim](query_vec, doc_vec))
                          for doc_id, doc_vec in enumerate(doc_vectors)
                          if doc_id + 1 != i]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])[:20]

    lines = ['| Rank | Doc # | Title | Similarity |',
             '|------|-------|-------|------------|']
    for rank, (doc_id, score) in enumerate(results_with_score, start=1):
        title = ' '.join(docs[doc_id - 1].title) or '(no title)'
        lines.append(f'| {rank} | {doc_id} | {title} | {score:.4f} |')

    return '\n'.join(lines)


if __name__ == '__main__':
    experiment()

    # print('# Top 20 Documents by Query')
    # for q in [6, 9, 22]:
    #     print(f'\n## Query {q}\n')
    #     for s in [True, False]:
    #         print(f'### Stem: {s}')
    #         print(query_top20_table(q, stem=s))
    #         print('\n#### Top 10 Documents Relevant Terms')
    #         print(query_top10_overlap(q, stem=s))

    # print('\n# Top 20 Documents by Document')
    # for d in [239, 1236, 2740]:
    #     print(f'\n## Document {d}\n')
    #     for s in [True, False]:
    #         print(f'\n### Stem: {s}')
    #         print(doc_top20_table(d, stem=s))