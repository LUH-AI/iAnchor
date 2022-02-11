import os

import numpy as np
import pytest
import sklearn
import sklearn.ensemble
from Anchor.anchor import Anchor
from Anchor.sampler import Tasktype

"""
Test funtions for tabular data anchor explainations
"""


@pytest.fixture(scope="session", autouse=True)
def setup():
    """
    For tabular tests we use the titanic dataset since its widely explored
    and the important features for explanations are known
    """
    data = np.genfromtxt("datasets/titanic.txt", delimiter=",")
    y_train = data[:, -1]
    X_train = data[:, :-1]
    c = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=5)
    c.fit(X_train, y_train)
    pytest.predict_fn = c.predict
    pytest.train_data = X_train


def test_tabular_greedy_search():
    explainer = Anchor(Tasktype.TABULAR)
    anchor = explainer.explain_instance(
        pytest.train_data[759].reshape(1, -1),
        pytest.predict_fn,
        "greedy",
        pytest.train_data,
        100,
        batch_size=32,
    )

    assert anchor.feature_mask == [2, 0, 8]
    assert np.isclose(anchor.coverage, 0.43)


def test_tabular_beam_search():
    explainer = Anchor(Tasktype.TABULAR)
    anchor = explainer.explain_instance(
        pytest.train_data[759].reshape(1, -1),
        pytest.predict_fn,
        "beam",
        pytest.train_data,
        100,
        batch_size=32,
    )

    assert anchor.feature_mask == [2, 0]
    assert np.isclose(anchor.coverage, 0.58)


def test_tabular_smac_search():
    explainer = Anchor(Tasktype.TABULAR)
    anchor = explainer.explain_instance(
        pytest.train_data[759].reshape(1, -1),
        pytest.predict_fn,
        "smac",
        pytest.train_data,
        100,
        batch_size=32,
    )  # 30 seconds test

    assert anchor.feature_mask == [0, 1, 2, 7, 8]
