from typing import Type, Any
from ianchor import Tasktype


class Visualizer:
    """
    Abstract Visualizer that is used as a factory for its
    subclasses. Use create(Tasktype) to initialise sub-
    classes for each task.
    """

    @staticmethod
    def create(type: Tasktype, *args: Any, **kwargs: Any) -> "Visualizer":
        """
        Creates the visualizer for the given task type.

        Parameters
        ----------
        type : Tasktype
            Type to create the visualizer for.

        Returns
        -------
        Visualizer
            The visualizer for the given task type.

        Raises
        ------
        ValueError
            If task type was not found.
        """
        visualizer: Type[Visualizer]
        if type == Tasktype.TABULAR:
            from ianchor.visualizers.tabular import TabularVisualizer

            visualizer = TabularVisualizer
        elif type == Tasktype.IMAGE:
            from ianchor.visualizers.image import ImageVisualizer

            visualizer = ImageVisualizer
        elif type == Tasktype.TEXT:
            from ianchor.visualizers.text import TextVisualizer

            visualizer = TextVisualizer
        else:
            raise ValueError("Unknown task type.")

        return visualizer(*args, **kwargs)
