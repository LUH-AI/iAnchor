import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from Anchor.anchor import Anchor, Tasktype
from Anchor.candidate import AnchorCandidate
from Anchor.sampler import Sampler
from Anchor.util import pytorch_image_wrapper
from PIL import Image
from skimage.data import astronaut
from skimage.segmentation import mark_boundaries, quickshift
from skimage.util import img_as_float
from torchvision.models import resnet18

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

    image = Image.open("../static/dog_paper.jpeg")
    input = preprocess(image).unsqueeze(0)

    pytest.predict_fn = model
    pytest.device = device
    pytest.input = input


def test_image_sampler():
    image = torch.Tensor(img_as_float(astronaut()[::2, ::2]))
    sampler = Sampler().create(Tasktype.IMAGE, image, pytest.predict_fn)
    candidate = AnchorCandidate(torch.arange(sampler._n_features))
    candidate, _ = sampler.sample(candidate, 3)

    assert candidate.n_samples == 3
    assert candidate.precision == 1


def test_image_greedy_search():
    explainer = Anchor(Tasktype.IMAGE)

    @pytorch_image_wrapper(pytest.device)
    def predict(x):
        return pytest.predict_fn(x)

    method_paras = {"desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        pytest.input.squeeze().permute(1, 2, 0),
        predict_fn=predict,
        method="greedy",
        method_specific=method_paras,
        num_coverage_samples=1000,
    )

    assert anchor.feature_mask == [11, 19]
    assert anchor.precision == 1.0


def test_image_beam_search():
    explainer = Anchor(Tasktype.IMAGE)

    @pytorch_image_wrapper(pytest.device)
    def predict(x):
        return pytest.predict_fn(x)

    method_paras = {"beam_size": 2, "desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        pytest.input.squeeze().permute(1, 2, 0),
        predict_fn=predict,
        method="beam",
        method_specific=method_paras,
        num_coverage_samples=1000,
    )

    assert anchor.feature_mask == [11, 19]
    assert anchor.precision == 1.0
