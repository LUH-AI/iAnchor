"""
Text Explanation
----------------

An example how to use iAnchor for text data.
"""

import os
import os.path
import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import spacy
import sys
from sklearn.feature_extraction.text import CountVectorizer

from ianchor import Tasktype
from ianchor.anchor import Anchor


# dataset from http://www.cs.cornell.edu/people/pabo/movie-review-data/
# Link: http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
def load_polarity(path="./datasets/rt-polaritydata/"):
    data = []
    labels = []
    f_names = ["rt-polarity.neg", "rt-polarity.pos"]
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), "rb"):
            try:
                line.decode("utf8")
            except Exception:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels


if __name__ == "__main__":

    # Prepare data
    data, labels = load_polarity()
    train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(
        train, train_labels, test_size=0.1, random_state=42
    )
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    val_labels = np.array(val_labels)

    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)
    test_vectors = vectorizer.transform(test)
    val_vectors = vectorizer.transform(val)

    # Get model
    c = sklearn.linear_model.LogisticRegression(random_state=1234)
    c.fit(train_vectors, train_labels)

    predictions = c.predict(val_vectors)

    print("Validation accuracy:", sklearn.metrics.accuracy_score(val_labels, predictions))

    def predict_lr(texts):
        return c.predict(vectorizer.transform(texts))

    spacy.cli.download("en_core_web_sm")  # Otherwise spacy can not load it
    nlp = spacy.load("en_core_web_sm")
    text_to_be_explained = "This is a good book."

    preprocessed_text = [word.text for word in nlp(text_to_be_explained)]

    # Get the explainer
    explainer = explainer = Anchor(Tasktype.TEXT)

    method_paras = {"beam_size": 1, "desired_confidence": 0.95}
    anchor = explainer.explain_instance(
        preprocessed_text,
        predict_lr,
        method="beam",
        method_specific=method_paras,
        num_coverage_samples=1,
    )

    visu = explainer.visualize(anchor, preprocessed_text)
    print(visu)
