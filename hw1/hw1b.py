import argparse
from itertools import groupby

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier # may be a good choice as well??
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import re

class SegmentClassifier:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def train(self, trainX, trainY, line_meta, doc_meta):
        self.clf = RandomForestClassifier()
        # Fit label encoder on all classes
        self.label_encoder.fit(trainY)

        # Extract features with previous class context
        X = []
        prev_class = None

        for x, y, meta in zip(trainX, trainY, line_meta):
            # Extract doc length and line position in doc
            doc, pos = meta
            doc_length = doc_meta[doc]

            features = self.extract_features(x, (doc_length, pos), prev_class)
            X.append(features)
            prev_class = y

        self.clf.fit(X, trainY)

    def extract_features(self, text, meta, prev=None):
        words = text.split()

        pattern1 = re.compile(r"\S+:")  # non-whitespace char followed by :
        pattern2 = re.compile(r"^[a-zA-Z]{0,3}>.+") # 0-3 characters followed by >
        pattern3 = re.compile(r"^(\d+[.)\-]|\(\d+\)) .+") # list item format: 1. 1) (1) 1-
        pattern4 = re.compile(r" {3,}") # 3+ spaces
        pattern5 = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.(com|org|edu)") # email
        pattern6 = re.compile(r"(\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4})") # phone number

        first_nonws = next((i for i, c in enumerate(text) if not c.isspace()), -1)
        first_alpha = next((i for i, c in enumerate(text) if c.isalpha()), -1)

        # Compute text midpoint
        stripped = text.strip()
        mid = len(stripped) // 2
        first_half, second_half = stripped[:mid], stripped[mid:]

        doc_length, pos = meta

        features = [
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            text.count(' '),

            sum(1 if w.isupper() else 0 for w in words),
            sum(1 if w.isalpha() else 0 for w in words),
            sum(1 if w.isnumeric() else 0 for w in words),

            sum(1 if not c.isalpha() and not c == ' ' else 0 for c in text),
            sum(1 if c.isalpha() else 0 for c in text),
            sum(1 if c.isnumeric() else 0 for c in text),

            1 if '@' in text else 0,
            1 if '"' in text else 0,
            sum(1 for c in text if c in ['/', '\\', '|', '_']),
            sum(1 for a, b in zip(text, text[1:]) if b == ',' and a.isnumeric()),
            sum(1 for a, b in zip(text, text[1:]) if b == ',' and a.isalpha()),
            1 if text.startswith(": ") else 0,

            1 if pattern1.match(words[0]) else 0,
            1 if pattern2.match(text.strip()) else 0,
            1 if pattern3.match(text.strip()) else 0,
            len(pattern4.findall(text)),
            1 if pattern5.search(text) else 0,
            1 if pattern6.search(text) else 0,

            sum(len(w) for w in words) / len(words) if words else 0,
            (first_alpha - first_nonws) if first_nonws != -1 and first_alpha != -1 else -1,

            sum(c.isalpha() for c in first_half) / len(first_half) if first_half else 0,
            sum(c.isalpha() for c in second_half) / len(second_half) if second_half else 0,

            -1 if prev is None else self.label_encoder.transform([prev])[0],
            
            pos / doc_length
        ]
        return features

    def classify(self, testX, line_meta, doc_meta):
        predictions = []
        prev_class = None
        
        for x, meta in zip(testX, line_meta):
            # Extract doc length and line position in doc
            doc, pos = meta
            doc_length = doc_meta[doc]

            features = self.extract_features(x, (doc_length, pos), prev_class)

            pred = self.clf.predict([features])[0]
            predictions.append(pred)

            prev_class = pred  # Update for next iteration
        
        return predictions



def load_data(file):
    with open(file) as fin:
        X = []
        y = []

        # Trackers to compute position metadata
        line_meta, doc_meta, doc, pos, prev = [], dict(), 0, 0, '#BLANK#'
        for line in fin:
            arr = line.strip().split('\t', 1)

            # If new document identified, set doc length and move to next doc
            if arr[0] == 'NNHEAD' and prev == '#BLANK#' and arr[1].lower().startswith('from'):
                doc_meta[doc] = pos
                doc += 1
                pos = 0

            prev = arr[0]   # Store previous line type for doc boundary detection
            if arr[0] == '#BLANK#':
                continue

            X.append(arr[1])
            y.append(arr[0])

            # Store line metadata and move to next line
            line_meta.append((doc, pos))
            pos += 1
        
        # Store length for last doc
        doc_meta[doc] = pos

        return X, y, line_meta, doc_meta


def lines2segments(trainX, trainY, line_meta):
    segX = []
    segY = []
    seg_meta = []   # Track doc positioning of segments (based on pos of first line)
    for y, group in groupby(zip(trainX, trainY, line_meta), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        items = list(group)
        x = '\n'.join(item[0].rstrip('\n') for item in items)
        segX.append(x)
        segY.append(y)
        seg_meta.append(items[0][2])    # Store segment position
    return segX, segY, seg_meta


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY, train_line_meta, train_doc_meta = load_data(args.train)
    testX, testY, test_line_meta, test_doc_meta = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY, train_line_meta = lines2segments(trainX, trainY, train_line_meta)
        testX, testY, test_line_meta = lines2segments(testX, testY, test_line_meta)

    classifier = SegmentClassifier()
    classifier.train(trainX, trainY, train_line_meta, train_doc_meta)
    outputs = classifier.classify(testX, test_line_meta, test_doc_meta)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()