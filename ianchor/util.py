import numpy as np


def pytorch_wrapper(device=None):
    """
    Decorator that converts anchor image samples
    (np.ndarray) to a Tensor and extracts the labels
    (with argmax)

    Use case:
        Adapter from Anchor Lib  -> predict_fn -> Anchor Lib

    Args:
        device ([type], optional): Pytorch device the model runs on.
                                    Defaults to torch.device("cpu").
    """

    import torch

    if device is None:
        device = torch.device("cpu")

    def _decorate(func):
        def wrapper(*args, **kwargs):
            x = torch.Tensor(args[0])
            x = x.to(device)
            y = func(x).cpu().detach().numpy()
            return np.argmax(y, axis=1)

        return wrapper

    return _decorate


def pytorch_image_wrapper(device=None):
    """
    Decorator that converts anchor image samples
    (np.ndarray) to a Tensor and extracts the labels
    (with argmax). Permuted the images to AxHxW before
    passing it to the wrapped method.

    Use case:
        Adapter from Anchor Lib  -> predict_fn -> Anchor Lib

    Args:
        device ([type], optional): Pytorch device the model runs on.
                                    Defaults to torch.device("cpu").
    """
    import torch

    if device is None:
        device = torch.device("cpu")

    def _decorate(func):
        def wrapper(x):
            x = torch.Tensor(x)
            x = x.to(device)
            y = func(x.permute(0, 3, 1, 2)).cpu().detach().numpy()
            return np.argmax(y, axis=1)

        return wrapper

    return _decorate
