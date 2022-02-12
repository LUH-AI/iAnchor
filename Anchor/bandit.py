import logging
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

from .candidate import AnchorCandidate
from .sampler import Sampler


@dataclass(frozen=True)
class KL_LUCB:
    """
    Multi armed bandit with lower and upper bound.
    Used to find the anchor rule with the expected highest precision.

    More information can be found in the following paper:
    http://proceedings.mlr.press/v30/Kaufmann13.pdf
    """

    # default values from original paper
    eps: float = 0.1
    delta: float = 0.1
    batch_size: int = 10
    verbose: bool = False

    # TODO: fix type annotations and implement this shit
    def get_best_candidates(
        self, candidates: list[AnchorCandidate], sampler: Sampler, top_n: int = 1,
    ):
        """
        Find top-n anchor candidates with highest expected precision.

        Args:
            candidates: list[AnchorCandidate]
            sampler: Sampler
            top_n: int
        Returns:
            best_candidates: list[AnchorCandidate]
        """

        assert len(candidates) > 0

        t = 1
        prec_ub = np.zeros(len(candidates))
        prec_lb = np.zeros(len(candidates))

        lt, ut, prec_lb, prec_ub = self.__update_bounds(
            candidates, prec_lb, prec_ub, t, top_n
        )
        prec_diff = prec_ub[ut] - prec_lb[lt]
        while prec_diff > self.eps:
            candidates[ut], _, _ = sampler.sample(candidates[ut], self.batch_size)
            candidates[lt], _, _ = sampler.sample(candidates[lt], self.batch_size)

            t += 1
            lt, ut, prec_lb, prec_ub = self.__update_bounds(
                candidates, prec_lb, prec_ub, t, top_n
            )
            prec_diff = prec_ub[ut] - prec_lb[lt]

        best_candidates_idxs = np.argsort([c.precision for c in candidates])[
            -top_n:
        ]  # use partioning

        return [candidates[idx] for idx in best_candidates_idxs]

    def __update_bounds(
        self,
        candidates: list[AnchorCandidate],
        lb: list[float],
        ub: list[float],
        t: int,
        top_n: int,
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Update current bounds for each candidate

        Args:
            candidates: list[AnchorCandidate]
            lb: list[float]
            ub: list[float]
            t: int
            top_n: int
        Returns:
            lt: int
            ut: int
        """

        means = [c.precision for c in candidates]  # mean precision per candidate
        sorted_means = np.argsort(means)

        beta = KL_LUCB.compute_beta(len(candidates), t, self.delta)
        j, nj = (
            sorted_means[-top_n:],
            sorted_means[:-top_n],
        )  # divide list into the top_n best candidates and the rest

        for f in j:
            lb[f] = KL_LUCB.dlow_bernoulli(
                means[f], beta / max(candidates[f].n_samples, 1)
            )
        for f in nj:
            ub[f] = KL_LUCB.dup_bernoulli(
                means[f], beta / max(candidates[f].n_samples, 1)
            )

        ut = nj[np.argmax(ub[nj])] if len(nj) != 0 else 0
        # candidate where upper bound of candidate is maximal
        lt = j[np.argmin(lb[j])]  # candidate where lower bound of candidate is minimal

        return lt, ut, lb, ub

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

        for _ in range(25):  # this should somehow converge?
            qm = (um + lm) / 2.0
            if KL_LUCB.kl_bernoulli(precision, qm) > level:
                um = qm
            # dont know why this should make sense at all?
            else:
                lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(precision: float, level: float):
        um = precision
        lm = max(min(1, precision - np.sqrt(level / 2.0)), 0)

        for _ in range(25):  # this should somehow converge?
            qm = (um + lm) / 2.0
            if KL_LUCB.kl_bernoulli(precision, qm) > level:
                lm = qm
            else:
                um = qm
        return lm

    @staticmethod
    def kl_bernoulli(precision: float, q: float):
        p = min(0.9999999999999999, max(0.0000001, precision))
        q = min(0.9999999999999999, max(0.0000001, q))

        return p * np.log(float(p) / q) + (1 - p) * np.log(float(1 - p) / (1 - q))

