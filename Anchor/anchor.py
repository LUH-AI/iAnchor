import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from skimage.segmentation import quickshift

from Anchor.bandit import KL_LUCB
from Anchor.candidate import AnchorCandidate
from Anchor.sampler import Sampler, Tasktype


@dataclass()
class Anchor:
    """
    Approach to explain predictions of a blackbox model using anchors.
    It returns the explaination with a precision and coverage score.

    More details can be found in the following paper:
    https://homes.cs.washington.edu/~marcotcr/aaai18.pdf
    """

    tasktype: Tasktype
    sampler: Sampler = field(init=False)
    verbose: bool = False
    coverage_data: np.array = field(init=False)

    def __post_init__(self):
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    def explain_instance(
        self,
        input: any,
        predict_fn: Callable[[any], np.array],
        method: str = "greedy",
        num_coverage_samples: int = 10000,
    ):
        self.kl_lucb = KL_LUCB()
        self.sampler = Sampler.create(self.tasktype, input, predict_fn)
        logging.info(" Start Sampling")
        _, self.coverage_data, _ = self.sampler.sample(
            AnchorCandidate(_feature_mask=[]), num_coverage_samples, False
        )
        exp = AnchorCandidate(_feature_mask=[])
        if method == "greedy":
            logging.info(" Start greedy search")
            exp = self.__greedy_anchor()
        # TODO add other methods

        return exp, self.sampler.segments

    def visualize():
        ...

    def generate_candidates(
        self, prev_anchors: list[AnchorCandidate], coverage_min: float
    ) -> list[AnchorCandidate]:
        new_candidates: list[AnchorCandidate] = []
        # iterate over possible features or predicates

        for feature in range(self.sampler.num_features):
            # check if we have no prev anchors and create a complete new set
            if len(prev_anchors) == 0:
                nc = AnchorCandidate(_feature_mask=[feature])
                new_candidates.append(nc)

            for anchor in prev_anchors:
                # check if feature already in the feature_mask of the anchor
                if feature in anchor.feature_mask:
                    continue

                # append new feature to candidate
                tmp = anchor.feature_mask.copy()
                tmp.append(feature)

                nc = AnchorCandidate(_feature_mask=tmp)
                nc.coverage = self.__calculate_coverage(nc)
                if nc.coverage > coverage_min:
                    new_candidates.append(nc)

        return new_candidates

    def __calculate_coverage(self, anchor: AnchorCandidate) -> float:
        included_samples = 0
        for mask in self.coverage_data:  # replace with numpy only
            # check if mask positive samples are included in the feature_mask of the anchor
            if np.all(np.isin(anchor.feature_mask, np.where(mask == 1)), axis=0):
                included_samples += 1

        return included_samples / self.coverage_data.shape[0]

    def check_valid_candidate(
        self,
        candidate: AnchorCandidate,
        beam_size: int = 1,
        sample_count: int = 10,
        delta: float = 0.05,
        dconf: float = 1,
        eps_stop: float = 0.05,
    ) -> bool:
        prec = candidate.precision
        beta = np.log(1.0 / (delta / (1 + (beam_size - 1) * self.sampler.num_features)))

        lb = KL_LUCB.dlow_bernoulli(prec, beta / candidate.n_samples)
        ub = KL_LUCB.dup_bernoulli(prec, beta / candidate.n_samples)

        while (prec >= dconf and lb < dconf - eps_stop) or (
            prec < dconf and ub >= dconf + eps_stop
        ):
            nc, _, _ = self.sampler.sample(candidate, sample_count)
            prec = nc.precision
            lb = KL_LUCB.dlow_bernoulli(prec, beta / nc.n_samples)
            ub = KL_LUCB.dup_bernoulli(prec, beta / nc.n_samples)

        return prec >= dconf and lb > dconf - eps_stop

    def __greedy_anchor(
        self,
        desired_confidence: float = 1,
        epsilon: float = 0.1,
        batch_size: int = 16,
        min_coverage: float = 0.2,
    ):
        """
        Greedy Approach to calculate the shortest anchor, which fullfills the precision constraint EQ3.
        """
        candidates = self.generate_candidates([], min_coverage)
        logging.info("test")
        anchor = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)[0]
        logging.info("test2")

        while anchor.precision < desired_confidence:
            candidates = self.generate_candidates([anchor], min_coverage)
            logging.info(candidates)
            anchor = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)[0]

        logging.info(anchor)
        return anchor

    def __beam_anchor():
        ...
