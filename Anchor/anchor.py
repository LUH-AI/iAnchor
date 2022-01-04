from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import torch
from skimage.segmentation import quickshift

from sampler import Sampler, Tasktype


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

    def explain_instance(self, input: any, predict_fn: Callable[[any], torch.Tensor]):
        self.sampler = Sampler.create(self.tasktype, input, predict_fn)
        exp = self.__greedy_anchor(self.sampler.sample)

    def __greedy_anchor(
        sample_fn: Callable,
        delta: float = 0.05,
        epsilon: float = 0.1,
        batch_size: int = 16,
    ):
        ...

    def beam_anchor():
        ...
