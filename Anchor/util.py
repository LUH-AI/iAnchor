import numpy as np

# import tensorflow as tf
import torch


# def tf_wrapper(func):
#     def wrapper(*args, **kwargs):
#         x = tf.convert_to_tensor(args[0])
#         y_proba = func(x).numpy()
#         func(*args, **kwargs)

#         return np.argmax(y_proba, dim=1)

#     return wrapper


# def pytorch_image_wrapper(torch_device):
#     def decorating(fn):
#         def inner(x):
#             torch_x = torch.Tensor(x).to(torch_device)
#             y_proba = fn(torch_x).numpy()
#             return np.argmax(y_proba, axis=1)

#         return inner

#     return decorating


def pytorch_wrapper(device=torch.device("cpu")):
    def _decorate(func):
        def wrapper(*args, **kwargs):
            x = torch.Tensor(args[0])
            x = x.to(device)
            y = func(x).cpu().detach().numpy()
            return np.argmax(y, axis=1)

        return wrapper

    return _decorate


def pytorch_image_wrapper(device=torch.device("cpu")):
    def _decorate(func):
        def wrapper(x):
            x = torch.Tensor(x)
            x = x.to(device)
            y = func(x.permute(0, 3, 1, 2)).cpu().detach().numpy()
            return np.argmax(y, axis=1)

        return wrapper

    return _decorate
