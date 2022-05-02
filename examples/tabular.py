"""
Tabular Explanation
-------------------

An example how to use iAnchor for tabular data.
"""

import numpy as np
import sklearn
import sklearn.ensemble
from ianchor.anchor import Anchor
from ianchor.samplers import Tasktype


if __name__ == "__main__":
    data = np.genfromtxt("datasets/titanic.txt", delimiter=",")
    y_train = data[:, -1]
    X_train = data[:, :-1]

    c = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=5, random_state=123)
    c.fit(X_train, y_train)
    print("Train accuracy:", sklearn.metrics.accuracy_score(y_train, c.predict(X_train)))

    explainer = Anchor(Tasktype.TABULAR)
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
    method_paras = {"beam_size": 1, "desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        input=X_train[759].reshape(1, -1),
        predict_fn=c.predict,
        method="beam",
        task_specific=task_paras,
        method_specific=method_paras,
        num_coverage_samples=100,
    )

    exit()

    visu = explainer.visualize(anchor, X_train[759])
    print(visu)
