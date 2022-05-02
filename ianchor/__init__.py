import datetime
from enum import Enum, auto

name = "iAnchor"
package_name = "ianchor"
author = "Kevin Schumann and Paul Heinemeyer"
author_email = "ks.kevinschumann@gmail.com"
description = "Reimplementation of Anchors: High-Precision Model-Agnostic Explanations."
url = "https://www.automl.org"
project_urls = {
    "Documentation": "https://LUH-AI.github.io/iAnchor/main",
    "Source Code": "https://github.com/LUH-AI/ianchor",
}
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, Kevin Schumann and Paul Heinemeyer"
version = "0.0.1"


class Tasktype(Enum):
    """
    Type of data that is going to be explained by the
    anchor.
    """

    TABULAR = auto()
    IMAGE = auto()
    TEXT = auto()
