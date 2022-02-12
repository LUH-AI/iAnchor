import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.segmentation import quickshift

from .candidate import AnchorCandidate
import matplotlib.pyplot as plt
import spacy


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


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
    def create(
        cls,
        type: Tasktype,
        input: any,
        predict_fn: Callable,
        task_specific: dict,
        **kwargs
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

        return cls.subclasses[type](
            input, predict_fn, **task_specific
        )  # every sampler needs input and predict function


class TabularSampler(Sampler):
    type: Tasktype = Tasktype.TABULAR

    def __init__(
        self,
        input: any,
        predict_fn: Callable[[any], np.array],
        dataset: any,
        column_names: list,
    ):
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
        self,
        candidate: AnchorCandidate,
        num_samples: int,
        calculate_labels: bool = True,
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        ...

        if self.dataset.shape[0] > num_samples:
            assert "Batch size must be smaller or equal to dataset rows."

        # pertubate
        sample_idxs = np.random.choice(
            self.dataset.shape[0], size=num_samples, replace=False
        )

        # fixiate feature mask
        samples = np.copy(self.dataset[sample_idxs])
        samples[:, candidate.feature_mask] = self.input[0, candidate.feature_mask]

        # calculate converage mask
        masks = (samples[:, :] != self.input).astype(int)

        if not calculate_labels:
            return None, masks, None

        # predict samples
        preds = self.predict_fn(samples)

        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, masks, None  # TODO remove third return variable


class ImageSampler(Sampler):
    """
    Image sampling with the help of superpixels.
    The original input image is permuated by switching off superpixel areas.

    More details can be found on the following website:
    https://www.oreilly.com/content/introduction-to-local-interpretable-model-agnostic-explanations-lime/
    """

    type: Tasktype = Tasktype.IMAGE

    def __init__(
        self, input: any, predict_fn: Callable[[any], np.array], dataset: any = None
    ):
        assert input.shape[2] == 3
        assert len(input.shape) == 3

        self.label = predict_fn(input[np.newaxis, ...])

        input = input.clone().cpu().detach().numpy()
        # run segmentation on the image
        self.features = quickshift(
            input.astype(np.double), kernel_size=4, max_dist=200, ratio=0.2
        )

        # parameters from original implementation
        segment_features = np.unique(self.features)
        self._n_features = len(segment_features)

        # create superpixel image by replacing superpixels by its mean in the original image
        self.sp_image = np.copy(input)
        for spixel in segment_features:
            self.sp_image[self.features == spixel, :] = np.mean(
                self.sp_image[self.features == spixel, :], axis=0
            )

        self.image = input
        self.predict_fn = predict_fn
        self.dataset = dataset

    def sample(
        self,
        candidate: AnchorCandidate,
        num_samples: int,
        calculate_labels: bool = True,
    ):
        data = np.random.randint(
            0, 2, size=(num_samples, self._n_features)
        )  # generate random feature mask for each sample
        data[:, candidate.feature_mask] = 1  # set present features to one

        if not calculate_labels:
            return None, data, None

        if self.dataset is not None:
            return self.sample_dataset(candidate, data, num_samples)
        else:
            return self.sample_mean_superpixel(candidate, data, num_samples)

    def sample_dataset(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        perturb_sample_idxs = np.random.choice(
            range(self.dataset.shape[0]), num_samples, replace=True
        )

        samples = np.stack(
            [
                self.__generate_image(mask, self.dataset[pidx])
                for mask, pidx in zip(data, perturb_sample_idxs)
            ],
            axis=0,
        )

        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data, self.features

    def sample_mean_superpixel(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
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
        samples = np.stack(
            [self.__generate_image(mask, self.sp_image) for mask in data], axis=0
        )

        preds = self.predict_fn(samples)

        # assert isinstance(
        #     preds, np.ndarray
        # ), "Result of your predict function should be of type numpy.ndarray"

        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data, self.features  # TODO remove third return variable

    def __generate_image(
        self, feature_mask: np.ndarray, perturb_image: np.ndarray
    ) -> np.array:
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
        mask = np.zeros(self.features.shape).astype(bool)
        for z in zeros:
            mask[self.features == z] = True
        img[mask] = perturb_image[mask]

        return img

    @property
    def num_features(self):
        return self._n_features


class TextSampler(Sampler):
    type: Tasktype = Tasktype.TEXT

    def __init__(
        self, input: any, predict_fn: Callable[[any], np.array], nlp_object: None
    ):

        if nlp_object is None:
            assert "Spacy object required for text anchor."

        self.label = predict_fn([input])
        self.input_processed = [word.text for word in nlp_object(input)]
        self.num_features = len(self.input_processed)

        self.predict_fn = predict_fn

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        self.bert = DistilBertForMaskedLM.from_pretrained("distilbert-base-cased")

        self.pr = {}
        for i in range(len(self.input_processed)):
            words = np.array(self.input_processed, dtype="|U80")
            words[i] = self.tokenizer.mask_token

            sentence = " ".join(words)
            w, p = self.prob(sentence)[0]
            self.pr[self.input_processed[i]] = min(
                0.5, dict(zip(w, p)).get(np.array(self.input_processed)[i], 0.01)
            )

    def prob(self, sentence):
        result = self.pred_topk_cbow(sentence)
        return [(a, exp_normalize(b)) for a, b in result]

    def pred_topk_cbow(self, sentence):
        encoded_text = self.tokenizer.encode(sentence, add_special_tokens=True)
        mask_token_idx = (
            np.array(encoded_text) == self.tokenizer.mask_token_id
        ).nonzero()[0]
        input = torch.tensor([encoded_text])

        with torch.no_grad():
            output = self.bert(input)[0]

        ret = []
        for i in mask_token_idx:
            v, top_preds = torch.topk(output[0, i], 500)
            words = self.tokenizer.convert_ids_to_tokens(top_preds)
            ret.append((words, v.numpy()))

        return ret

    def sample(
        self,
        candidate: AnchorCandidate,
        num_samples: int,
        calculate_labels: bool = True,
    ):
        feature_masks = np.zeros((num_samples, len(self.input_processed)))
        for idx, word in enumerate(self.input_processed):
            # for every word decide if we should mask it or not
            if idx in candidate.feature_mask:
                continue

            prob = self.pr[word]
            feature_masks[:, idx] = np.random.choice(
                [0, 1], num_samples, p=[1 - prob, prob]
            )

        # unmask words in candidate mask
        feature_masks[:, candidate.feature_mask] = 1

        if not calculate_labels:
            return None, feature_masks, None

        return self.__sample_pertubated_sentences(candidate, feature_masks, num_samples)

    def __generate_sentence(self, feature_mask: np.ndarray) -> str:
        # Generate new sentences by masking words according to the
        # feature mask. Then new words are samples. This is done
        # done word for word

        # mask words given the feature mask
        sentence_cp = np.array(self.input_processed, dtype="|U80")
        sentence_cp[feature_mask != 1] = self.tokenizer.mask_token

        # sample new word for each word
        masked_word = np.where(feature_mask == 0)[0]
        for word_idx in masked_word:
            mod_sentence = " ".join(sentence_cp)
            words, probs = self.prob(mod_sentence)[0]
            sentence_cp[word_idx] = np.random.choice(words, p=probs)

        feature_mask = sentence_cp == np.array(self.input_processed, dtype="|U80")

        return " ".join(sentence_cp)

    def __sample_pertubated_sentences(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        sentences = np.apply_along_axis(self.__generate_sentence, 1, data).reshape(
            -1, 1
        )

        # predict pertubed sentences
        preds = self.predict_fn(sentences.flatten().tolist())
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data, None
