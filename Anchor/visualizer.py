from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

from .candidate import AnchorCandidate
from .sampler import Tasktype


class Visualizer:
    """
    Abstract Visualizer that is used as a factory for its
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
    def create(cls, type: Tasktype, **kwargs):
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
            **kwargs
        )  # every sampler needs input and predict function


class ImageVisualizer(Visualizer):
    """
    Visalizer for image anchors.
    """

    type: Tasktype = Tasktype.IMAGE

    def visualize(
        self, anchor: AnchorCandidate, original_instance: np.array, features: np.array
    ):
        """
        Visualizes the image anchor

        Args:
            anchor (AnchorCandidate): AnchorCandiate which feature masks is used to explain the instance.
            original_instance (np.array): (M, N[, 3]) image that is going to explained.
            features (np.array): Segments of the original image.

        Returns:
            (np.ndarray): (M, N, 3) array of floats. 
            An image in which the boundaries between labels are superimposed on the original image.
        """
        idxs = np.argwhere(~np.isin(features, anchor.feature_mask))
        mask = features.copy()
        mask[idxs[:, 0], idxs[:, 1]] = 0
        exp_visu = mark_boundaries(
            original_instance, mask, mode="thick", outline_color=(0, 0, 0)
        )

        return exp_visu


class TextVisualizer(Visualizer):
    """
    Visualizer for text anchors. 
    """

    type: Tasktype = Tasktype.TEXT

    def visualize(self, anchor: AnchorCandidate, original_instance: str, features: any):
        """
        Visualizes the text anchor.

        Args:
            anchor (AnchorCandidate): AnchorCandiate which feature masks is used to explain the instance
            original_instance (str): Text to be explained
            features (np.array): Unused

        Returns:
            (str): Returns the orignial sentence with the importants words marked in yellow. 
        """
        explanation = []
        for i, word in enumerate(original_instance):
            if i in anchor.feature_mask:
                explanation.append("\033[93m" + word + "\033[0m")
            else:
                explanation.append(word)

        return " ".join(explanation)


class TabularVisualizer(Visualizer):
    """
    Visalizer for tabular anchors.
    """

    type: Tasktype = Tasktype.TABULAR

    def visualize(
        self, anchor: AnchorCandidate, original_instance: np.array, features: np.array
    ):
        """
        Visualizes the tabular anchor.

        Args:
            anchor (AnchorCandidate): AnchorCandiate which feature masks is used to explain the instance
            original_instance (str): Tabular instance (row) that is to be explained.
            features (np.array): Columns names of the dataset.

        Returns:
            (str): Returns the orignial sentence with the importants words marked in yellow. 
        """

        exp_visu = [
            f"{k} = {v}"
            for i, (k, v) in enumerate(zip(features, original_instance))
            if i in anchor.feature_mask
        ]

        return " AND ".join(exp_visu)
