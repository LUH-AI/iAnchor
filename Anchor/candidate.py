from dataclasses import dataclass, field

import torch


@dataclass()
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    _feature_mask: torch.Tensor
    _precision: float = 0
    _n_samples: int = 0
    _positive_samples: int = 0

    def update(self, positives, n_samples):
        self._n_samples += n_samples
        self._positive_samples += positives
        self._precision = self._positive_samples / self._n_samples

    @property
    def feature_mask(self):
        return self._feature_mask

    @property
    def precision(self):
        return self._precision

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def positive_samples(self):
        return self._positive_samples

