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

    More information can be found in the following paper:
    http://proceedings.mlr.press/v30/Kaufmann13.pdf
    """

    eps: float = field()
    delta: float = field()
    batch_size: int = field()
    verbose: bool = field()

    # TODO: fix type annotations and implement this shit
    def get_best_candidates(
        self, candidates: list[AnchorCandidate], sampler, top_n: int = 1,
    ):
        """
        Find top-n anchor candidates with highest expected precision.
        """
        t = 1
        prec_ub = torch.zeros(len(candidates))
        prec_lb = torch.zeros(len(candidates))

        lt, ut = self.__update_bounds(candidates, prec_lb, prec_ub, t)

        while (prec_ub - prec_lb) > self.delta:
            pass

        # @TODO Decide what should be returned. Could be the top n indices or the means of all candidates.

    def __update_bounds(
        candidates: list[AnchorCandidate], lb: list[float], ub: list[float], t: int
    ) -> Tuple[list[float], list[float]]:
        """
        Updates current bounds
        """
        means = [c.precision for c in candidates]  # mean precision per candidate
        # @TODO Implement

    # Following part is completely based on the original implementation, since there is not much one could optimize or change

    @staticmethod
    def compute_beta(n_features: int, t: int, delta: float):
        alpha = 1.1  # constant from paper
        k = 405.5  # constant from paper
        temp = np.log(k * n_features * (t ** alpha) / delta)

        return temp + np.log(temp)

    @staticmethod
    def dup_bernoulli(precision: float, level: float):
        lm = precision
        um = min(min(1, precision + np.sqrt(level / 2.0)), 1)
        qm = (um + lm) / 2.0

        if KL_LUCB.kl_bernoulli(precision, qm) > level:
            um = qm
        # dont know why this should make sense at all?
        # else:
        #     lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(precision: float, level: float):
        um = precision
        lm = max(min(1, precision - np.sqrt(level / 2.0)), 0)
        qm = (um + lm) / 2.0

        if KL_LUCB.kl_bernoulli(precision, qm) > level:
            lm = qm
        # else:
        #     um = qm
        return lm

    @staticmethod
    def kl_bernoulli(precision: float, q: float):
        p = min(0.9999999999999999, max(0.0000001, precision))
        q = min(0.9999999999999999, max(0.0000001, q))

        return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))

