from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from skimage.segmentation import quickshift

from Anchor.bandit import KL_LUCB
from Anchor.candidate import AnchorCandidate
from Anchor.sampler import Sampler, Tasktype


@dataclass(frozen=True)
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

    def explain_instance(
        self,
        input: any,
        predict_fn: Callable[[any], torch.Tensor],
        method: str,
        num_coverage_samples: int,
    ):
        self.kl_lucb = KL_LUCB()
        self.sampler = Sampler.create(self.tasktype, input, predict_fn)
        _, self.coverage_data, _ = self.sampler.sample(
            AnchorCandidate(_feature_mask=[]), num_coverage_samples
        )
        exp = AnchorCandidate([])
        if method == "greedy":
            exp = self.__greedy_anchor()
        # TODO add other methods

        return exp

    def generate_candidates(
        self,
        prev_anchors: list[AnchorCandidate],
        coverage_min: float,
        num_coverage_samples: int,
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
                anchor.append_feature(feature)
                coverage = self.__calculate_coverage(anchor, num_coverage_samples)
                if coverage > coverage_min:
                    new_candidates.append(anchor)

        return new_candidates

    def __calculate_coverage(self, anchor: AnchorCandidate) -> float:
        included_samples = 0
        for mask in self.coverage_data:
            # check if mask positive samples are included in the feature_mask of the anchor
            if np.all(np.isin(np.where(mask == True), anchor.feature_mask), axis=0):
                included_samples += 1

        return included_samples / self.coverage_data.shape[0]

    def __greedy_anchor(
        self,
        delta: float = 0.1,
        epsilon: float = 0.1,
        batch_size: int = 16,
        min_coverage: float = 0.5,
        num_coverage_samples: int = 10000,
    ):
        """
        Greedy Approach to calculate the shortest anchor, which fullfills the precision constraint EQ3.
        """
        candidates = self.generate_candidates([], min_coverage, num_coverage_samples)
        best_idx = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)
        anchor = candidates[best_idx]

        while anchor.precision < 1 - delta:
            candidates = self.generate_candidates(
                [anchor], min_coverage, num_coverage_samples
            )
            best_idx = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)
            anchor = candidates[best_idx]

        return anchor

    def beam_anchor():
        ...
