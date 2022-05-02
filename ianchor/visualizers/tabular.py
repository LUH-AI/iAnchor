import numpy as np

from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.visualizers import Visualizer


class TabularVisualizer(Visualizer):
    """
    Visalizer for tabular anchors.
    """

    type: Tasktype = Tasktype.TABULAR

    def visualize(self, anchor: AnchorCandidate, original_instance: np.array, features: np.array):
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
