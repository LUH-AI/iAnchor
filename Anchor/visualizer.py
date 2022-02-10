from .candidate import AnchorCandidate
from .sampler import Tasktype
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

class Visualizer():
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
    def create(
        cls, type: Tasktype, **kwargs
    ):
        """
        Creates subclass depending on typ.

        Args:
            typ: Tasktype
        Returns:
            Subclass that is used for the given Tasktype.
        """
        if type not in cls.subclasses:
            raise ValueError("Bad message type {}".format(type))

        return cls.subclasses[type](**kwargs)  # every sampler needs input and predict function

class ImageVisualizer(Visualizer):
    type: Tasktype = Tasktype.IMAGE

    def visualize(self, anchor: AnchorCandidate, original_image: np.array, features: np.array):
        idxs = np.argwhere(~np.isin(features, anchor.feature_mask))
        mask = features.copy()
        mask[idxs[:, 0], idxs[:, 1]] = 0
        img = mark_boundaries(original_image, mask, mode="thick", outline_color=(0, 0, 0))

        plt.imshow(img)
        

class TextVisualizer(Visualizer):
    type: Tasktype = Tasktype.TEXT

    def visualize(self, anchor: AnchorCandidate):
        ...
    
class TabularVisualizer(Visualizer):
    type: Tasktype = Tasktype.TABULAR

    def visualize(self, anchor: AnchorCandidate):
        ...