from collections import Counter, defaultdict
from math import e

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import sys, os
from typing import List, NamedTuple, Dict, Tuple

_orig_cwd = os.getcwd()

stemmer = SnowballStemmer('english')

def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

# Implement a nearest centroid classifier (also called a Rocchio classifier)

class Sentence(NamedTuple):
    id: int
    text: List[str]
    label: int
    key_idx: int

def read_docs(file: str, stem=False, stop=False, lr_tokens=False) -> List[Sentence]:
    '''
    1. Initialize the “document” vectors, where each example sentence is its 
    own document. The weighting options will be discussed below. In addition,
    store the classification label for later use.
    '''
    sentences = []
    with open(file) as f:
        for line in f:
            # read line
            parts = line.strip().split('\t')
            doc_id, label, raw_text = int(parts[0]), int(parts[1]), parts[2]
            raw_words = raw_text.split()

            # strip keyword marker and tokenize
            marked = next((w for w in raw_words if w.startswith('.X-')), None)
            keyword = marked[3:].lower() if marked else None
            cleaned = raw_text.replace(marked, keyword, 1) if marked else raw_text
            words = [w.lower() for w in word_tokenize(cleaned)]

            # optionally remove stopwords
            if stop:
                words = [w for w in words if w not in stopwords]

            # find keyword index (before stemming so keyword matches as-is)
            key_idx = next((i for i, w in enumerate(words) if w == keyword), -1)

            # optionally stem
            if stem:
                words = [stemmer.stem(w) for w in words]

            # make words immediately adjacent to keyword special tokens
            if lr_tokens:
                if key_idx > 0:
                    words[key_idx - 1] = f'L_{words[key_idx - 1]}'
                if key_idx < len(words) - 1:
                    words[key_idx + 1] = f'R_{words[key_idx + 1]}'

            sentences.append(Sentence(doc_id, words, label, key_idx))
    return sentences


### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]) -> float:
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x: Dict[str, float], y: Dict[str, float]) -> float:
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


### Weighting Methods

def compute_unweighted(s: Sentence) -> Dict[str, float]:
    vec = defaultdict(float)
    for i, word in enumerate(s.text):
        if i == s.key_idx: continue
        vec[word] += 1
    return dict(vec)  # convert back to a regular dict

def compute_exponential_weighting(s: Sentence) -> Dict[str, float]:
    '''
    Smooth exponential distance decay, where a word’s weight is
    1 / (distance of the word to the target word).
    '''
    vec = defaultdict(float)
    for i, word in enumerate(s.text):
        if i == s.key_idx: continue
        # compute exponential decay
        vec[word] += 1 / abs(s.key_idx - i)
    return dict(vec)  # convert back to a regular dict

def compute_stepped_weighting(s: Sentence) -> Dict[str, float]:
    '''
    Stepped weighting, where adjacent words are given weight 6.0, words 
    2-3 away are given weight 3.0, and all other words are given weight 1.0.
    '''
    vec = defaultdict(float)
    for i, word in enumerate(s.text):
        if i == s.key_idx: continue
        # compute stepped weighting
        weight = 1
        dist = abs(s.key_idx - i)
        if dist == 1: weight = 6
        elif dist <= 3: weight = 3
        vec[word] += weight
    return dict(vec)  # convert back to a regular dict

def compute_custom_weighting(s: Sentence, sigma=2) -> Dict[str, float]:
    '''
     Weighting scheme of your own choice: Gaussian decay
     Based on a normal distribution curve
     '''
    vec = defaultdict(float)
    for i, word in enumerate(s.text):
        if i == s.key_idx: continue
        dist = abs(s.key_idx - i)
        # compute gaussian decay
        vec[word] += e ** (- (dist ** 2) / (2 * (sigma ** 2)))
    return dict(vec)  # convert back to a regular dict


def compute_doc_freqs(sentences: List[Sentence]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for s in sentences:
        words = set()
        for word in s.text:
            words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tfidf(vec: Dict[str, float], doc_freqs: Dict[str, int], N: int) -> Dict[str, float]:
    '''
    Computes the tf-idf vector for a sentence based on some weighted vector representation.
    '''
    tfidf_computed = dict()

    # compute tf-idf for each term
    for word, tf in vec.items():
        if tf == 0 or doc_freqs.get(word, 0) == 0: continue
        tfidf_computed[word] = (1 + np.log10(tf)) * np.log10(N / doc_freqs[word])

    return tfidf_computed


def create_profile_vectors(sentences: List[Sentence], vecs: List[Dict[str, float]]) -> Tuple[Dict, Dict]:
    '''
    2. Using the vectors in the training set, create two profile vectors V_profile1
    and V_profile2, where V_profilei is the average (or centroid) of all of the
    training vectors labelled as sense i.
    '''
    tally = {1: defaultdict(float), 2: defaultdict(float)}
    counts = {1: 0, 2: 0}

    for s, vec in zip(sentences, vecs):
        for word, weight in vec.items():
            tally[s.label][word] += weight
        counts[s.label] += 1

    profile1 = {w: v / counts[1] for w, v in tally[1].items()}
    profile2 = {w: v / counts[2] for w, v in tally[2].items()}
    return profile1, profile2


def compute_similarity(x: Dict[str, float], profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
    '''
    3. For each vector in the test set, compute its similarity to each profile vector:
    If sim1 >= sim2, then label the vector as sense 1, otherwise sense 2. 
    For ease of evaluation, you may also wish to print sim1 - sim2 and sort by 
    this value. Large positive numbers will indicate examples that are strongly 
    sense 1, large negative numbers will indicate examples that are strongly
    sense 2, and values close to 0 are examples that are ambiguous.
    '''
    sim1 = cosine_sim(x, profile1)
    sim2 = cosine_sim(x, profile2)
    diff = sim1 - sim2
    return 1 if diff >= 0 else 2


def run(train_file: str, test_file: str, stem=False, stop=False, lr_tokens=False, tfidf=False):
    '''
    4. In Step 3, keep a running count of the total number of the test examples 
    that your program classifies correctly and incorrectly. At the end, print 
    out the percent correct: total correct / (total correct + total incorrect).
    '''
    # setup processing
    train_sentences = read_docs(train_file, stem, stop, lr_tokens)
    test_sentences = read_docs(test_file, stem, stop, lr_tokens)
    train_vecs = [compute_unweighted(s) for s in train_sentences]
    test_vecs = [compute_unweighted(s) for s in test_sentences]

    if tfidf:
        N = len(train_sentences)
        doc_freqs = compute_doc_freqs(train_sentences)
        train_vecs = [compute_tfidf(v, doc_freqs, N) for v in train_vecs]
        test_vecs = [compute_tfidf(v, doc_freqs, N) for v in test_vecs]

    profile1, profile2 = create_profile_vectors(train_sentences, train_vecs)

    # tally correct categorizations
    correct = 0
    incorrect = 0
    for s, vec in zip(test_sentences, test_vecs):
        sense = compute_similarity(vec, profile1, profile2)
        if sense == s.label:
            correct += 1 
        else:
            incorrect += 1

    return correct / (correct + incorrect)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise Exception("Usage: python hw3.py <file-train.tsv> <file-dev.tsv>")
    train_file = os.path.join(_orig_cwd, sys.argv[1])
    test_file = os.path.join(_orig_cwd, sys.argv[2])
    acc = run(train_file, test_file, stem=False, stop=True, lr_tokens=True, tfidf=True)
    print(f'Accuracy: {acc}')
