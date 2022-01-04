from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np
import torch

from candidate import AnchorCandidate


@dataclass(frozen=True)
class KL_LUCB:
    """
    Multi armed bandit with lower and upper bound.
    Used to find the anchor rule with the expected highest precision.
    """

    eps: float = field()
    delta: float = field()
    batch_size: int = field()
    verbose: bool = field()

    # TODO: fix type annotations and implement this shit
    def get_best_candidates(
        self, canditates: list[AnchorCandidate], sampler, top_n: int = 1,
    ):
        prec_ub = torch.zeros(len(canditates))
        prec_lb = torch.zeros(len(canditates))

        while (prec_ub - prec_lb) > self.delta:
            pass

    def __update_bounds(
        canditates: list[AnchorCandidate], ub: list[float], lb: list[float], t: int
    ) -> Tuple[list[float], list[float]]:
        """
        Updates current bounds
        """
        ...

    @staticmethod
    def compute_beta(n_features: int, t: int, delta: float):
        ...

    @staticmethod
    def dup_bernoulli(precision: float, level: float):
        ...

    @staticmethod
    def dlow_bernoulli(precision: float, level: float):
        ...

    @staticmethod
    def kl_bernoulli(precision: float, q: float):
        p = min(0.9999999999999999, max(0.0000001, precision))
        q = min(0.9999999999999999, max(0.0000001, q))
        return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))

