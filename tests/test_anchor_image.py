import matplotlib.pyplot as plt
import pytest
import torch
import torchvision.transforms.functional as TF
from Anchor.candidate import AnchorCandidate
from Anchor.sampler import Sampler, Tasktype
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
    pytest.predict_fn = model


def test_image_sampler():
    image = torch.Tensor(img_as_float(astronaut()[::2, ::2]))
    sampler = Sampler().create(Tasktype.IMAGE, image, pytest.predict_fn)
    candidate = AnchorCandidate([])
    candidate, segments = sampler.sample(candidate, 3)

    assert (
        len(torch.unique(segments)) == 44
    )  # dependent on segmentation method and hyperparameters

    fig = plt.figure()
    plt.imshow(mark_boundaries(image, segments.numpy()))
    plt.show()

