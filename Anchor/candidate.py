from dataclasses import dataclass, field


@dataclass()
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    _feature_mask: list
    _precision: float = 0
    _n_samples: int = 0
    _positive_samples: int = 0
    _coverage: float = 0

    def update_precision(self, positives: int, n_samples: int):
        self._n_samples += n_samples
        self._positive_samples += positives
        self._precision = self._positive_samples / self._n_samples

    def append_feature(self, feature: int):
        self._feature_mask.append(feature)

    @property
    def feature_mask(self) -> list:
        return self._feature_mask

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def positive_samples(self) -> int:
        return self._positive_samples

    @property
    def coverage(self) -> float:
        return self._coverage

    @coverage.setter
    def coverage(self, val):
        self._coverage = val

