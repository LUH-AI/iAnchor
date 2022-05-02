from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.visualizers import Visualizer


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
