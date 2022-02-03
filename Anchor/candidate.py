from dataclasses import dataclass


@dataclass()
class AnchorCandidate:
    """
    Reprensents a possible candidate in the process of finding the best anchor.
    """

    feature_mask: list
    precision: float = 0
    n_samples: int = 0
    positive_samples: int = 0
    coverage: float = 0

    def update_precision(self, positives: int, n_samples: int):
        self.n_samples += n_samples
        self.positive_samples += positives
        self.precision = self.positive_samples / self.n_samples

    def append_feature(self, feature: int):
        self.feature_mask.append(feature)

