from typing import Any, Callable, Dict, Type

from ianchor import Tasktype


class Sampler:
    """
    Abstract Sampler that is used as a factory for its
    subclasses. Use create(Tasktype) to initialise sub-
    classes for each task.
    """

    @staticmethod
    def create(type: Tasktype, input: Any, predict_fn: Callable, task_specific: Dict) -> "Sampler":
        """
        Creates the sampler for the given task type.

        Parameters
        ----------
        type : Tasktype
            Task type to create the sampler for.
        input : Any
            Input data.
        predict_fn : Callable
            Prediction function.
        task_specific : Dict
            Arguments for the sampler.

        Returns
        -------
        Sampler
            The sampler for the given task type.

        Raises
        ------
        ValueError
            If task type was not found.
        """
        sampler: Type[Sampler]
        if type == Tasktype.TABULAR:
            from ianchor.samplers.tabular import TabularSampler

            sampler = TabularSampler
        elif type == Tasktype.IMAGE:
            from ianchor.samplers.image import ImageSampler

            sampler = ImageSampler
        elif type == Tasktype.TEXT:
            from ianchor.samplers.text import TextSampler

            sampler = TextSampler
        else:
            raise ValueError("Unknown task type.")

        return sampler(input, predict_fn, **task_specific)  # type: ignore
