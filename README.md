<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="static/logo/logo.png" alt="Logo">
  </a>
</div>

#

<!-- ABOUT THE PROJECT -->
## About The Project
An interpretable and easy-to-understand python version of the Anchor explanation method from [Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf). Our implementation is inspired by [this code](https://github.com/marcotcr/anchor). Furthermore, it supports optimization with the SMAC optimizer besides KL-divergence. The code is unit tested and the tests can be run with pytest.


<!-- GETTING STARTED -->
## Getting Started
This section describes how to get started explaining your own black-box models.

### Prerequisites
To install the required packages we recommend using [Conda](https://docs.conda.io/en/latest/). Our used environment can be easily installed with conda.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/automl-classroom/iml-ws21-projects-risingnumpygods.git
   ```
2. Install conda environment
   ```sh
   conda env create -f iAnchor
   ```
3. Activate the environment
   ```sh
   conda activate iAnchor
   ```

### Tests
1. Go to the project directory and run
   ```sh
    pytest tests/*
   ```

<!-- USAGE EXAMPLES -->
## Usage

We provided an example jupyter notebook for different use cases. You can find the notebooks [here](/notebooks/). The notebooks cover the following use cases:
* Explanation of image prediction
* Explanation of tabular prediction
* Explanation of text prediction

### Example
Suppose you want to explain a tabular instance prediction. You can get an explanation with a few lines of code.
```py
import numpy as np
import sklearn
from Anchor.anchor import Anchor, Tasktype

# Load dataset
data = np.genfromtxt('../datasets/titanic.txt', delimiter=',')
y_train = data[:, -1]
X_train = data[:, :-1]

# Train classifier
c = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=5, random_state=123)
c.fit(X_train, y_train)
print('Train', sklearn.metrics.accuracy_score(y_train, c.predict(X_train)))

# Init explainer for specific task
explainer = Anchor(Tasktype.TABULAR)

# Get explanation for desired instance
task_paras = {"dataset": X_train, "column_names": ["PcClass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]}
method_paras = {"beam_size": 1, "desired_confidence": 1.0}
anchor = explainer.explain_instance(
    input=X_train[759].reshape(1, -1),
    predict_fn=c.predict,
    method="beam",
    task_specific=task_paras,
    method_specific=method_paras,
    num_coverage_samples=100,
)

# Visualize explanation
visu = explainer.visualize(anchor, X_train[759])
print(visu)
```

_For more advanced usage and architecture insights you can look at the [docs](/docs/)_.



<!-- CONTACT -->
## Contact

1. [Kevin Schumann](https://github.com/kevin-schumann) - ks.kevinschumann@gmail.com
2. [Paul Heinemeyer](https://github.com/SwiftPredator) - paul_heinemeyer@outlook.de


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Anchors](https://github.com/marcotcr/anchor)
* [SMAC3](https://github.com/automl/SMAC3)



