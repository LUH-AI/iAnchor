import os

import numpy as np
import pytest
import sklearn
import sklearn.ensemble
import spacy
from Anchor.anchor import Anchor
from Anchor.sampler import Tasktype
from sklearn.feature_extraction.text import CountVectorizer

"""
Test funtions for text data anchor explainations
"""


def load_polarity(path="/datasets/rt-polaritydata/"):
    """
    Helper function to preprocess polarity dataset
    """
    data = []
    labels = []
    f_names = ["rt-polarity.neg", "rt-polarity.pos"]
    for (l, f) in enumerate(f_names):
        for line in open(os.path.join(path, f), "rb"):
            try:
                line.decode("utf8")
            except:
                continue
            data.append(line.strip())
            labels.append(l)
    return data, labels


@pytest.fixture(scope="session", autouse=True)
def setup():
    """
    For text tests we use cornell polarity data and a simple sample sentence
    """
    nlp = spacy.load("en_core_web_sm")
    text_to_be_explained = "This is a good book ."
    preprocessed_text = [word.text for word in nlp(text_to_be_explained)]
    data, labels = load_polarity()

    train, _, train_labels, _ = sklearn.model_selection.train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    train, _, train_labels, _ = sklearn.model_selection.train_test_split(
        train, train_labels, test_size=0.1, random_state=42
    )
    train_labels = np.array(train_labels)

    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(train)
    train_vectors = vectorizer.transform(train)

    c = sklearn.linear_model.LogisticRegression()
    c.fit(train_vectors, train_labels)

    pytest.predict_fn = lambda x: c.predict(vectorizer.transform(x))
    pytest.input = preprocessed_text


def test_text_greedy_search():
    explainer = Anchor(Tasktype.TABULAR)
    method_paras = {"desired_confidence": 0.95}
    anchor = explainer.explain_instance(
        input=pytest.input,
        predict_fn=pytest.predict_fn,
        method="greedy",
        method_specific=method_paras,
        num_coverage_samples=1,
    )

    assert anchor.feature_mask == [2, 0]
    assert np.isclose(anchor.coverage, 0.54)


def test_text_beam_search():
    explainer = Anchor(Tasktype.TABULAR)

    method_paras = {"beam_size": 2, "desired_confidence": 0.95}
    anchor = explainer.explain_instance(
        input=pytest.input,
        predict_fn=pytest.predict_fn,
        method="beam",
        method_specific=method_paras,
        num_coverage_samples=1,
    )

    assert anchor.feature_mask == [0, 2]
    assert np.isclose(anchor.coverage, 0.54)
