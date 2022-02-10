import logging

from Anchor.visualizer import Visualizer

logging.basicConfig(level=logging.INFO)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
from skimage.segmentation import quickshift

from Anchor.bandit import KL_LUCB
from Anchor.candidate import AnchorCandidate
from Anchor.sampler import Sampler, Tasktype

from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter
from .visualizer import Visualizer


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
    visualizer: Visualizer = field(init=False)
    verbose: bool = False
    coverage_data: np.array = field(init=False)

    def __post_init__(self):
        logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

    def explain_instance(
        self,
        input: any,
        predict_fn: Callable[[any], np.array],
        method: str = "greedy",
        dataset: any = None,
        num_coverage_samples: int = 10000,
        desired_confidence: float = 1,
        epsilon: float = 0.15,
        batch_size: int = 16,
        beam_size=4,
        verbose=False,
    ):
        self.kl_lucb = KL_LUCB(eps=epsilon, batch_size=batch_size, verbose=verbose)
        self.sampler = Sampler.create(self.tasktype, input, predict_fn, dataset)
        self.batch_size = batch_size
        logging.info(" Start Sampling")
        _, self.coverage_data, _ = self.sampler.sample(
            AnchorCandidate(feature_mask=[]), num_coverage_samples, False
        )
        exp = AnchorCandidate(feature_mask=[])
        if method == "greedy":
            logging.info(" Start Greedy Search")
            exp = self.__greedy_anchor()
        elif method == "beam":
            logging.info(" Start Beam Search")
            exp = self.__beam_anchor(
                desired_confidence=desired_confidence, beam_size=beam_size,
            )
        elif method == "smac":
            logging.info(" Start SMAC Search")
            exp = self.__smac_anchor()

        return exp

    def visualize(self, anchor: AnchorCandidate, instance: np.ndarray):
        Visualizer.create(self.tasktype).visualize(anchor, instance, self.sampler.features)

    def generate_candidates(
        self, prev_anchors: list[AnchorCandidate], coverage_min: float
    ) -> list[AnchorCandidate]:
        new_candidates: list[AnchorCandidate] = []
        # iterate over possible features or predicates

        for feature in range(self.sampler.num_features):
            # check if we have no prev anchors and create a complete new set
            if len(prev_anchors) == 0:
                nc = AnchorCandidate(feature_mask=[feature])
                new_candidates.append(nc)

            for anchor in prev_anchors:
                # check if feature already in the feature_mask of the anchor
                if feature in anchor.feature_mask:
                    continue

                # append new feature to candidate
                tmp = anchor.feature_mask.copy()
                tmp.append(feature)

                nc = AnchorCandidate(feature_mask=tmp)
                nc.coverage = self.__calculate_coverage(nc)
                if nc.coverage >= coverage_min:
                    new_candidates.append(nc)

        return new_candidates

    def __calculate_coverage(self, anchor: AnchorCandidate) -> float:
        included_samples = 0
        for mask in self.coverage_data:  # replace with numpy only
            # check if mask positive samples are included in the feature_mask of the anchor
            if np.all(np.isin(anchor.feature_mask, np.where(mask == 1)), axis=0):
                included_samples += 1

        return included_samples / self.coverage_data.shape[0]

    def __check_valid_candidate(
        self,
        candidate: AnchorCandidate,
        beam_size: int,
        sample_count: int,
        dconf: float,
        delta: float = 0.1,
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

        # print(lb, ub, prec)
        return prec >= dconf and lb > dconf - eps_stop

    def __greedy_anchor(
        self, desired_confidence: float = 1, min_coverage: float = 0.2,
    ):
        """
        Greedy Approach to calculate the shortest anchor, which fullfills the precision constraint EQ3.
        """
        candidates = self.generate_candidates([], min_coverage)
        anchor = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)[0]

        while not self.__check_valid_candidate(
            anchor, 1, self.batch_size, desired_confidence
        ):
            candidates = self.generate_candidates([anchor], min_coverage)
            logging.info(candidates)
            anchor = self.kl_lucb.get_best_candidates(candidates, self.sampler, 1)[0]

        logging.info(anchor)
        return anchor

    def __beam_anchor(
        self, desired_confidence: float, beam_size: int,
    ):

        max_anchor_size = self.sampler.num_features
        current_anchor_size = 1
        best_of_size = {0: []}  # A0
        best_candidate = AnchorCandidate([])  # A*

        while current_anchor_size < max_anchor_size:
            # Generate candidates
            candidates = self.generate_candidates(
                best_of_size[current_anchor_size - 1], best_candidate.coverage,
            )
            if len(candidates) == 0:
                break
            best_candidates = self.kl_lucb.get_best_candidates(
                candidates, self.sampler, min(beam_size, len(candidates))
            )
            best_of_size[current_anchor_size] = best_candidates

            for c in best_candidates:
                if (
                    self.__check_valid_candidate(
                        c,
                        beam_size=beam_size,
                        sample_count=self.batch_size,
                        dconf=desired_confidence,
                    )
                    and c.coverage > best_candidate.coverage
                ):
                    best_candidate = c

            current_anchor_size += 1

        return best_candidate

    def __smac_anchor(self,):
        # create config space
        configspace = ConfigurationSpace()

        # mask the possible features
        for i in range(self.sampler.num_features):
            configspace.add_hyperparameter(UniformIntegerHyperparameter(str(i), 0, 1))

        # create Szenario
        scenario = Scenario(
            {
                "run_obj": "quality",
                "algo_runs_timelimit": 1 * 30,
                "cs": configspace,
                "deterministic": "true",  # each config gets evaluated once, other option would be to track candidates and average precision / coverage
            }
        )

        # create optimizer
        smac = SMAC4BB(
            scenario=scenario,
            tae_runner=self.smac_optimize,
            rng=np.random.RandomState(42),  # TODO change to global seed
        )
        best_mask = (
            smac.optimize()
        )  # TODO should also return found precision and coverage - Maybe we can get this to return the full candidate
        # return candidate
        feature_mask = [int(f_idx) for f_idx, mv in best_mask.items() if mv]
        stats = smac.runhistory.data[
            next(reversed(smac.runhistory.data))
        ].additional_info

        return AnchorCandidate(
            feature_mask=feature_mask,
            precision=stats["precision"],
            coverage=stats["coverage"],
        )

    def smac_optimize(self, config):
        feature_mask = [int(f_idx) for f_idx, mv in config.items() if mv]
        # create candidate from config which is the feature mask to evaluate
        candidate = AnchorCandidate(feature_mask)
        # calculate expected precision
        candidate, _, _ = self.sampler.sample(candidate, self.batch_size)
        candidate.coverage = self.__calculate_coverage(candidate)

        info = {"precision": candidate.precision, "coverage": candidate.coverage}

        return ((1 - candidate.precision) + (1 - candidate.coverage)) / 2, info

