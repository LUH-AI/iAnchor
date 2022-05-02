from typing import Any, Callable, Tuple

import numpy as np
import torch
from transformers import DistilBertForMaskedLM, DistilBertTokenizer

from ianchor import Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.samplers import Sampler


class TextSampler(Sampler):
    """
    TextSampler generates new text instances
    given an AnchorCandidate by fixiating the
    candidates features and replacing masked
    words with alternatives given from bert
    model.
    """

    type: Tasktype = Tasktype.TEXT

    def __init__(self, input: Any, predict_fn: Callable[[Any], np.array]):
        """
        Initialises TextSampler with the given
        predict_fn, input, dataset and nlp_object

        Predict_fn will be used to predict all the
        samples and the input.

        Args:
            input (list(str)): Sentences as list of tokens.
            predict_fn (Callable[[Any], np.array]): Black box model predict function.
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

        def exp_normalize(x):
            b = x.max()
            y = np.exp(x - b)
            return y / y.sum()

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
        mask_token_idx = (np.array(encoded_text) == self.tokenizer.mask_token_id).nonzero()[0]

        # predict top 500 for each masked word
        preds_per_word = []
        for i in mask_token_idx:
            v, top_preds = torch.topk(output[0, i], 500)
            words = self.tokenizer.convert_ids_to_tokens(top_preds)
            preds_per_word.append((words, v.numpy()))

        return preds_per_word

    def sample(
        self, candidate: AnchorCandidate, num_samples: int, calculate_labels: bool = True,
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
            feature_masks[:, idx] = np.random.choice([0, 1], num_samples, p=[1 - prob, prob])

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
        sentences = np.apply_along_axis(self.__generate_sentence, 1, data).reshape(-1, 1)

        # predict pertubed sentences
        preds = self.predict_fn(sentences.flatten().tolist())
        labels = (preds == self.label).astype(int)

        # update candidate
        candidate.update_precision(np.sum(labels), num_samples)
        return candidate, data
