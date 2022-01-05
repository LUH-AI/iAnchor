from dataclasses import dataclass, field

import torch


@dataclass
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    feature_mask: torch.Tensor = field()

    precision: float = 0
    n_samples: int = 0
    positive_samples: int = 0

    @property
    def feature_mask(self):
        return self.feature_mask

    @property
    def precision(self):
        return self.precision

    @property
    def n_samples(self):
        return self.n_samples

    @property
    def positive_samples(self):
        return self.positive_samples
