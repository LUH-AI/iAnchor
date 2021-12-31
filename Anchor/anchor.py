from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Protocol

import torch
from skimage.segmentation import quickshift


class Tasktype(Enum):
    """
    Type of data that is going to be explained by the
    anchor.
    """

    TABULAR = auto()
    IMAGE = auto()
    TEXT = auto()


class Sampler:
    """
    Abstract Sampler that is used as a factory for its 
    subclasses. Use create(Tasktype) to initialise sub-
    classes for each task.
    """

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        """
        Registers every subclass in the subclass-dict.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.type] = cls

    @classmethod
    def create(cls, typ: Tasktype):
        """
        Creates subclass depending on typ.

        Args:
            typ: Tasktype 
        Returns:
            Subclass that is used for the given Tasktype.
        """
        if typ not in cls.subclasses:
            raise ValueError("Bad message type {}".format(typ))

        return cls.subclasses[typ]()


class TabularSampler(Sampler):
    typ: Tasktype = Tasktype.TABULAR

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> torch.Tensor:
        ...


class ImageSampler(Sampler):
    typ: Tasktype = Tasktype.IMAGE

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> torch.Tensor:
        ...


class TextSampler(Sampler):
    typ: Tasktype = Tasktype.TEXT

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> torch.Tensor:
        ...


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

    def __post_init__(self):
        self.sampler = Sampler.create(self.tasktype)

    def explain_instance(self, input: any, predict_fn: Callable[[any], torch.Tensor]):
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
