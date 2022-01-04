from dataclasses import dataclass

import torch


@dataclass
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    precision: float = 0

    @property
    def precision(self):
        return self.precision
