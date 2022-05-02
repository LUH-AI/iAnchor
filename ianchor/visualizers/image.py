import numpy as np
from skimage.segmentation import mark_boundaries

from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.visualizers import Visualizer


class ImageVisualizer(Visualizer):
    """
    Visalizer for image anchors.
    """

    type: Tasktype = Tasktype.IMAGE

    def visualize(self, anchor: AnchorCandidate, original_instance: np.array, features: np.array):
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
        exp_visu = mark_boundaries(original_instance, mask, mode="thick", outline_color=(0, 0, 0))

        return exp_visu
