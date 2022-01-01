from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

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
    def create(cls, type: Tasktype):
        """
        Creates subclass depending on typ.

        Args:
            typ: Tasktype 
        Returns:
            Subclass that is used for the given Tasktype.
        """
        if type not in cls.subclasses:
            raise ValueError("Bad message type {}".format(type))

        return cls.subclasses[type]()


class TabularSampler(Sampler):
    type: Tasktype = Tasktype.TABULAR

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> torch.Tensor:
        ...


class ImageSampler(Sampler):
    type: Tasktype = Tasktype.IMAGE

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        label = torch.argmax(predict_fn(input.permute(2, 0, 1).unsqueeze(0))[0])
        # run segmentation on the image
        segments = torch.from_numpy(
            quickshift(input.double(), kernel_size=4, max_dist=200, ratio=0.2)
        )  # parameters not from original implementation
        segment_features = torch.unique(segments)
        n_features = len(segment_features)

        # create superpixel image by replacing superpixels by its mean in the original image
        sp_image = torch.clone(input)
        for spixel in segment_features:
            sp_image[segments == spixel, :] = torch.mean(
                sp_image[segments == spixel, :], axis=0
            )
        return segments, None


class TextSampler(Sampler):
    type: Tasktype = Tasktype.TEXT

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
