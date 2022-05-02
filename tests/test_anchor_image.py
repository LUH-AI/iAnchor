import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.data import astronaut
from skimage.segmentation import mark_boundaries, quickshift
from skimage.util import img_as_float
from torchvision.models import resnet18

from ianchor.anchor import Anchor, Tasktype
from ianchor.candidate import AnchorCandidate
from ianchor.sampler import Sampler
from ianchor.util import pytorch_image_wrapper

"""
Test funtions for the image sampler and anchor explainations on images
"""


@pytest.fixture(scope="session", autouse=True)
def setup():
    model = resnet18(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open("./static/dog_paper.jpeg")
    input = preprocess(image).unsqueeze(0)

    @pytorch_image_wrapper(device)
    def predict(x):
        return model(x)

    pytest.predict_fn = predict
    pytest.device = device
    pytest.input = input


def test_image_sampler():
    sampler = Sampler().create(
        Tasktype.IMAGE,
        pytest.input.squeeze().permute(1, 2, 0),
        pytest.predict_fn,
        task_specific={},
    )
    candidate = AnchorCandidate(torch.arange(sampler.num_features))
    candidate, _ = sampler.sample(candidate, 3)

    assert candidate.n_samples == 3
    assert candidate.precision == 1


def test_image_greedy_search():
    explainer = Anchor(Tasktype.IMAGE)

    method_paras = {"desired_confidence": 0.8}
    anchor = explainer.explain_instance(
        pytest.input.squeeze().permute(1, 2, 0),
        predict_fn=pytest.predict_fn,
        method="greedy",
        method_specific=method_paras,
        num_coverage_samples=1000,
    )

    assert anchor.feature_mask == [19]
    assert anchor.precision >= 0.8


def test_image_beam_search():
    explainer = Anchor(Tasktype.IMAGE)

    method_paras = {"beam_size": 2, "desired_confidence": 0.8}
    anchor = explainer.explain_instance(
        pytest.input.squeeze().permute(1, 2, 0),
        predict_fn=pytest.predict_fn,
        method="beam",
        method_specific=method_paras,
        num_coverage_samples=1000,
    )

    assert anchor.feature_mask == [11, 4]
    assert anchor.precision >= 0.8
