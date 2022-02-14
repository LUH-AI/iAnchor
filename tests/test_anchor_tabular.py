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
    c = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100, n_jobs=5, random_state=123
    )
    c.fit(X_train, y_train)
    task_paras = {
        "dataset": X_train,
        "column_names": [
            "PcClass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ],
    }
    pytest.predict_fn = c.predict
    pytest.train_data = X_train
    pytest.task_paras = task_paras


def test_tabular_greedy_search():
    explainer = Anchor(Tasktype.TABULAR)
    method_paras = {"desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        input=pytest.train_data[759].reshape(1, -1),
        predict_fn=pytest.predict_fn,
        method="greedy",
        task_specific=pytest.task_paras,
        method_specific=method_paras,
        num_coverage_samples=100,
        batch_size=32,
    )

    assert anchor.feature_mask == [2, 0]
    assert np.isclose(anchor.coverage, 0.54)


def test_tabular_beam_search():
    explainer = Anchor(Tasktype.TABULAR)

    method_paras = {"beam_size": 2, "desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        input=pytest.train_data[759].reshape(1, -1),
        predict_fn=pytest.predict_fn,
        method="beam",
        task_specific=pytest.task_paras,
        method_specific=method_paras,
        num_coverage_samples=100,
        batch_size=32,
    )

    assert anchor.feature_mask == [0, 2]
    assert np.isclose(anchor.coverage, 0.54)


"""
This is not recommended since the result is dependant on the users hardware
and takes really long to run if runtime is set to inf.

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

"""
