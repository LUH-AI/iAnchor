"""
Image Explanation
-----------------

An example how to use iAnchor for image data.
"""

import torch
import torchvision.transforms as transforms
from ianchor import Tasktype
from ianchor.anchor import Anchor
from ianchor.util import pytorch_image_wrapper
from torchvision.models import inception_v3
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # model: c, h, w
    # numpy, sklearn matplotlib: w, h, c
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open("static/dog.jpeg")
    input = preprocess(image).unsqueeze(0)  # b, c, h, w

    # Get the model
    model = inception_v3(pretrained=True)
    model.eval()

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare predict function
    @pytorch_image_wrapper(device)
    def predict(x):
        return model(x)

    # Get our explainer
    explainer = Anchor(Tasktype.IMAGE)

    # And explain instance
    method_paras = {"beam_size": 2, "desired_confidence": 1.0}
    anchor = explainer.explain_instance(
        input.squeeze().permute(1, 2, 0),
        predict_fn=predict,
        method="beam",
        method_specific=method_paras,
        num_coverage_samples=1000,
    )

    visu = explainer.visualize(anchor, input.squeeze().permute(1, 2, 0))
    plt.imshow(visu)
