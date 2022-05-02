from typing import Any, Callable, List, Tuple

import numpy as np

from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.samplers import Sampler


class TabularSampler(Sampler):
    """
    TabularSampler generates new tabular instances
    given an AnchorCandidate by fixiating the
    candidates features and sampling random values
    within the dataset.
    """

    type: Tasktype = Tasktype.TABULAR

    def __init__(
        self, input: Any, predict_fn: Callable[[Any], np.array], dataset: Any, column_names: List,
    ):
        """
        Initialises TabularSampler with the given
        predict_fn, input, dataset and column names.

        Predict_fn will be used to predict all the
        samples and the input.

        Args:
            input (any): Tabular row that is to be explained.
            predict_fn (Callable[[any], np.array]): Black box model predict function.
            dataset (any): Tabular dataset from which samples will be collected. Expected to be discretized.
            column_names (list): Columns names of the dataset.
        """
        if dataset is None:
            assert "Dataset must be given for tabular explaination."
        if column_names is None:
            assert "Column names must be given for tabular explaination."

        self.predict_fn = predict_fn
        self.input = input
        self.label = predict_fn(input)
        self.dataset = dataset
        self.features = column_names
        self.num_features = self.dataset.shape[1]

        assert (
            len(column_names) == self.num_features
        ), "column_names length must match dataset column dimension."

    def sample(
        self, candidate: AnchorCandidate, num_samples: int, calculate_labels: bool = True,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Generates num_samples samples by choosing random values
        out of self.dataset and setting the self.input features
        that are withing the candidates feature mask.

        Args:
            candidate (AnchorCandidate): AnchorCandiate which contains the features to be fixated.
            num_samples (int): Number of samples that shall be generated.
            calculate_labels (bool, optional): When true label of the samples will predicted. In that case the
            candiates precision will be updated. Defaults to True.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Structure: [AnchorCandiate, coverage_mask]. In case
            calculate_labels is False return [None, coverage_mask].
        """

        if self.dataset.shape[0] > num_samples:
            assert "Batch size must be smaller or equal to dataset rows."

        # pertubate
        sample_idxs = np.random.choice(self.dataset.shape[0], size=num_samples, replace=False)

        # fixiate feature mask
        samples = np.copy(self.dataset[sample_idxs])
        samples[:, candidate.feature_mask] = self.input[0, candidate.feature_mask]

        # calculate converage mask
        masks = (samples[:, :] != self.input).astype(int)

        if not calculate_labels:
            return None, masks

        # predict samples
        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, masks
