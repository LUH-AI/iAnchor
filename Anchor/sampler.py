import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import numpy as np
import torch
from skimage.segmentation import quickshift

from .candidate import AnchorCandidate


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
    def create(cls, type: Tasktype, input: any, predict_fn: Callable, **kwargs):
        """
        Creates subclass depending on typ.

        Args:
            typ: Tasktype
        Returns:
            Subclass that is used for the given Tasktype.
        """
        if type not in cls.subclasses:
            raise ValueError("Bad message type {}".format(type))

        return cls.subclasses[type](
            input, predict_fn, **kwargs
        )  # every sampler needs input and predict function


class TabularSampler(Sampler):
    type: Tasktype = Tasktype.TABULAR

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        ...


class ImageSampler(Sampler):
    """
    Image sampling with the help of superpixels.
    The original input image is permuated by switching off superpixel areas.

    More details can be found on the following website:
    https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/
    """

    type: Tasktype = Tasktype.IMAGE
    device: torch.device = torch.device("cpu")

    def __init__(self, input: any, predict_fn: Callable[[any], np.array], **kwargs):
        assert input.shape[2] == 3
        assert len(input.shape) == 3

        self.label = np.argmax(
            predict_fn(input.permute(2, 0, 1).unsqueeze(0)).cpu().detach().numpy(),
            axis=1,
        )

        input = input.cpu().detach().numpy()
        # run segmentation on the image
        self.segments = quickshift(
            input.astype(np.double), kernel_size=4, max_dist=200, ratio=0.2
        )

        # parameters from original implementation
        segment_features = np.unique(self.segments)
        self._n_features = len(segment_features)

        # create superpixel image by replacing superpixels by its mean in the original image
        self.sp_image = np.copy(input)
        for spixel in segment_features:
            self.sp_image[self.segments == spixel, :] = np.mean(
                self.sp_image[self.segments == spixel, :], axis=0
            )
        self.image = input
        self.predict_fn = predict_fn

    def sample(
        self, candidate: AnchorCandidate, num_samples: int
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        """
        Sample function for image data.
        Generates random image samples from the distribution around the original image.

        Args:
            candidate: AnchorCandidate
            num_samples: int
        Returns:
            candidate: AnchorCandidate
        """
        data = np.random.randint(
            0, 2, size=(num_samples, self._n_features)
        )  # generate random feature mask for each sample
        data[:, candidate.feature_mask] = 1  # set present features to one
        samples = np.stack([self.__generate_image(mask) for mask in data], axis=0)
        input = torch.Tensor(samples).to(self.device)
        preds = self.predict_fn(input.permute(0, 3, 1, 2)).cpu().detach().numpy()
        preds_max = np.argmax(preds, axis=1)
        labels = (preds_max == self.label).astype(int)
        # print(self.label, preds_max)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data, self.segments
        # retur

    def __generate_image(self, feature_mask: np.ndarray) -> np.array:
        """
        Generate sample image given some feature mask.
        The true image will get permutated dependent on the feature mask.
        Pixel which are outmasked by the mask are replaced by the corresponding superpixel pixel.

        Args:
            feature_mask: np.ndarray
        Returns:
            permutated image: np.array
        """
        img = self.image.copy()
        zeros = np.where(feature_mask == 0)[0]
        mask = np.zeros(self.segments.shape).astype(bool)
        for z in zeros:
            mask[self.segments == z] = True
        img[mask] = self.sp_image[mask]

        return img

    @property
    def num_features(self):
        return self._n_features


class TextSampler(Sampler):
    type: Tasktype = Tasktype.TEXT

    def sample(
        self, input: any, predict_fn: Callable[[any], torch.Tensor]
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        ...
