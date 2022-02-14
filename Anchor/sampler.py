import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
from skimage.segmentation import quickshift
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

from .candidate import AnchorCandidate


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
    """
    TabularSampler generates new tabular instances
    given an AnchorCandidate by fixiating the
    candidates features and sampling random values
    within the dataset.
    """

    type: Tasktype = Tasktype.TABULAR

    def __init__(
        self,
        input: any,
        predict_fn: Callable[[any], np.array],
        dataset: any,
        column_names: list,
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
        self,
        candidate: AnchorCandidate,
        num_samples: int,
        calculate_labels: bool = True,
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
        sample_idxs = np.random.choice(
            self.dataset.shape[0], size=num_samples, replace=False
        )

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
        """
        Initialises ImageSampler with the given
        predict_fn, input and image dataset.

        Predict_fn will be used to predict all the
        samples and the input.

        When dataset equals None samples are generated
        by utilising mean superpixels.

        Args:
            input (any): Image that is to be explained.
            predict_fn (Callable[[any], np.array]): Black box model predict function.
            dataset (any): Image dataset from which samples will be collected
        """

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
        self.num_features = len(segment_features)

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
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Generates num_samples samples by choosing random values
        out of self.dataset and setting the self.input features
        that are withing the candidates feature mask.

        When dataset is None then samples are generated by
        utilizing the mean superpixel else the image datset
        is sampled

        Args:
            candidate (AnchorCandidate): AnchorCandiate which contains the features to be fixated.
            num_samples (int): Number of samples that shall be generated.
            calculate_labels (bool, optional): When true label of the samples will predicted. In that case the
                candiates precision will be updated. Defaults to True.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Structure: [AnchorCandiate, coverage_mask]. In case
            calculate_labels is False return [None, coverage_mask].
        """
        data = np.random.randint(
            0, 2, size=(num_samples, self.num_features)
        )  # generate random feature mask for each sample
        data[:, candidate.feature_mask] = 1  # set present features to one

        if not calculate_labels:
            return None, data

        # generate either samples from the dataset or mean superpixel
        if self.dataset is not None:
            return self.sample_dataset(candidate, data, num_samples)
        else:
            return self.sample_mean_superpixel(candidate, data, num_samples)

    def sample_dataset(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Samples num_samples samples by utilising the image dataset.

        Args:
            candidate (AnchorCandidate): AnchorCandidate which precision will be updated.
            data (np.ndarray): Features masks
            num_samples (int): Number of samples to be generated.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Structure: [AnchorCandiate, coverage_mask]
        """
        perturb_sample_idxs = np.random.choice(
            range(self.dataset.shape[0]), num_samples, replace=True
        )

        # generate samples from the dataset
        samples = np.stack(
            [
                self.__generate_image(mask, self.dataset[pidx])
                for mask, pidx in zip(data, perturb_sample_idxs)
            ],
            axis=0,
        )

        # predict samples
        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate prec
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data

    def sample_mean_superpixel(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Sample function for image data.
        Generates random image samples from the distribution around the original image.

        Args:
            candidate (AnchorCandidate): AnchorCandidate which precision will be updated.
            data (np.ndarray): Generated feature mask.
            num_samples (int): Number of samples to be generated.

        Returns:
            Tuple[AnchorCandidate, np.ndarray]: Returns the AnchorCandiate and the feature masks.
        """
        # Sample function for image data.
        # Generates random image samples from the distribution around the original image.

        # Args:
        #     candidate (AnchorCandidate)
        #     num_samples (int)
        # Returns:
        #     candidate (AnchorCandidate)
        # """
        samples = np.stack([self.__generate_image(mask) for mask in data], axis=0)

        # predict labels
        preds = self.predict_fn(samples)
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)

        return candidate, data

    def __generate_image(self, feature_mask: np.ndarray) -> np.array:
        """
        Generate sample image given some feature mask.
        The true image will get permutated dependent on the feature mask.
        Pixel which are outmasked by the mask are replaced by the corresponding superpixel pixel.

        Args:
            feature_mask (np.ndarray): Feature mask to generate picture with 

        Returns:
            np.array: Generated image.
        """
        img = self.image.copy()
        zeros = np.where(feature_mask == 0)[0]
        mask = np.zeros(self.features.shape).astype(bool)
        for z in zeros:
            mask[self.features == z] = True
        img[mask] = self.sp_image[mask]

        return img


class TextSampler(Sampler):
    """
    TextSampler generates new text instances
    given an AnchorCandidate by fixiating the
    candidates features and replacing masked
    words with alternatives given from bert
    model.
    """

    type: Tasktype = Tasktype.TEXT

    def __init__(self, input: any, predict_fn: Callable[[any], np.array]):
        """
        Initialises TextSampler with the given
        predict_fn, input, dataset and nlp_object

        Predict_fn will be used to predict all the
        samples and the input.

        Args:
            input (list(str)): Sentences as list of tokens.
            predict_fn (Callable[[any], np.array]): Black box model predict function.
        """
        self.label = predict_fn([" ".join(input)])
        self.input = input
        self.num_features = len(self.input)
        self.predict_fn = predict_fn

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")
        self.bert = DistilBertForMaskedLM.from_pretrained("distilbert-base-cased")

        # contains top500k probability of each word in the input
        self.pr = {}

        # caches bert predictions
        self.prob_cache = {}

        # mask each word separetly and predict topk given its context
        for i in range(len(self.input)):
            masked_sentence = self.input.copy()
            masked_sentence[i] = self.tokenizer.mask_token

            sentence = " ".join(masked_sentence)

            w, p = self.prob(sentence)[0]
            self.pr[self.input[i]] = min(0.5, dict(zip(w, p)).get(self.input[i], 0.01))

    def prob(self, sentence: str):
        """
        Given a senteces with masked tokens predicts
        the cbow (word alternatives, exp normalized
        probabilites) for each word.

        Args:
            sentence (str): Sentences which maskes word alternatives
                            and probabilites shall be predicted.

        Returns:
            results (list(tuple(str, float)))
        """
        if sentence in self.prob_cache:
            return self.prob_cache[sentence]

        result = self.pred_topk_cbow(sentence)
        normalized_result = [(a, exp_normalize(b)) for a, b in result]

        self.prob_cache[sentence] = normalized_result
        return normalized_result

    def pred_topk_cbow(self, sentence):
        """
        Give a sentences with masked tokens predict
        alternative words (and the corresponding probabilities)
        via bert.

        For each masked token returns the top500 predictions
        with their probabilities.

        Returns:
            predictions (list(tuple(str, float)))
        """
        # encode text
        encoded_text = self.tokenizer.encode(sentence, add_special_tokens=True)
        input = torch.tensor([encoded_text])

        with torch.no_grad():
            output = self.bert(input)[0]

        # get idx for mask token
        mask_token_idx = (
            np.array(encoded_text) == self.tokenizer.mask_token_id
        ).nonzero()[0]

        # predict top 500 for each masked word
        preds_per_word = []
        for i in mask_token_idx:
            v, top_preds = torch.topk(output[0, i], 500)
            words = self.tokenizer.convert_ids_to_tokens(top_preds)
            preds_per_word.append((words, v.numpy()))

        return preds_per_word

    def sample(
        self,
        candidate: AnchorCandidate,
        num_samples: int,
        calculate_labels: bool = True,
    ) -> Tuple[AnchorCandidate, np.ndarray]:
        """
        Generates num_samples samples by choosing if words
        that are not within the candiates feature mask should
        be masked given their original probability (self.pr).

        Args:
            candidate (AnchorCandidate): AnchorCandiate which contains the features to be fixated.
            num_samples (int): Number of samples that shall be generated.
            calculate_labels (bool, optional): When true label of the samples will predicted. In that case the
            candiates precision will be updated. Defaults to True.

        Returns:
            Tuple[AnchorCandidate, np.ndarray, np.ndarray]: Structure: [AnchorCandiate, coverage_mask, None]. In case
            calculate_labels is False return [None, coverage_mask, None].
        """
        feature_masks = np.zeros((num_samples, len(self.input)))
        for idx, word in enumerate(self.input):
            if idx in candidate.feature_mask:
                continue

            # decide if we should mask the word or not
            prob = self.pr[word]
            feature_masks[:, idx] = np.random.choice(
                [0, 1], num_samples, p=[1 - prob, prob]
            )

        # unmask words in candidate mask
        feature_masks[:, candidate.feature_mask] = 1

        if not calculate_labels:
            return None, feature_masks

        return self.__sample_pertubated_sentences(candidate, feature_masks, num_samples)

    def __generate_sentence(self, feature_mask: np.ndarray) -> str:
        """
        Generate new sentence by masking words according to the
        feature mask. For each masked word new words are samples.
        This is done word for word in an iterative manner to generate
        more coherent sentences.

        Args:
            feature_mask (np.ndarray): Features mask where != 1 denotes
                                        that a word shall be masked

        Returns:
            str: Generated sentence
        """
        # Generate new sentences by masking words according to the
        # feature mask. Then new words are samples. This is done
        # done word for word

        # mask words given the feature mask
        sentence_cp = np.array(self.input, dtype="|U80")
        sentence_cp[feature_mask != 1] = self.tokenizer.mask_token

        # sample new word for each word
        masked_word = np.where(feature_mask == 0)[0]
        for word_idx in masked_word:
            mod_sentence = " ".join(sentence_cp)
            words, probs = self.prob(mod_sentence)[0]
            sentence_cp[word_idx] = np.random.choice(words, p=probs)

        feature_mask = sentence_cp == np.array(self.input, dtype="|U80")

        return " ".join(sentence_cp)

    def __sample_pertubated_sentences(
        self, candidate: AnchorCandidate, data: np.ndarray, num_samples: int,
    ) -> Tuple[AnchorCandidate, np.ndarray, np.ndarray]:
        """
        Generate num_sampels new sentences (via self.__generate_sentence),
        predicts the labels and updates the precision for the AnchorCandidate
        candidate.

        Args:
            candidate (AnchorCandidate): Candidate for which the samples will be generated for.
                                            This candiates precisision will be updated in the process.
            data (np.ndarray): Several feature_masks. For each mask a new sentence will be generated.
            num_samples (int): Number of samples. Used to calculate the precision.

        Returns:
            Tuple[AnchorCandidate, np.ndarray, np.ndarray]: Structure [AnchorCandidate, feature_masks, None]
        """
        sentences = np.apply_along_axis(self.__generate_sentence, 1, data).reshape(
            -1, 1
        )

        # predict pertubed sentences
        preds = self.predict_fn(sentences.flatten().tolist())
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)
        return candidate, data
