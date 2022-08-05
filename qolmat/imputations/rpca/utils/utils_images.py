"""
General utility functions for images
"""

import numpy as np
import tqdm
from PIL import Image


def corrupt_image(image: np.ndarray, ratio: float) -> np.ndarray:
    """Add some noise to an image (3D array)

    Parameters
    ----------
    image : np.ndarray
        image to be corrupted
    ratio : np.ndarray
        ratio in [0,1] of pixels to be corrupted

    Returns
    -------
    np.ndarray
        corrupted image
    """

    d1, d2, d3 = image.shape
    to_corrupt = int(d1 * d2 * ratio)
    corrupted = np.copy(image)
    for i in range(d3):
        layer = image[:, :, i]
        indices = np.random.choice(np.arange(d1 * d2), to_corrupt, replace=False)
        values = np.random.randint(0, 255 + 1, to_corrupt)
        result = np.ravel(layer)
        result[indices] = values
        corrupted[:, :, i] = np.reshape(result, (d1, d2))

    return corrupted


def similarity_images(im1: np.ndarray, im2: np.ndarray) -> float:
    """Compute a similarity measure between two images

    Parameters
    ----------
    im1 : np.ndarray
        first image
    im2 : np.ndarray
        second image

    Returns
    -------
    float
        similarity score
    """

    centeredA = im1 - np.mean(im1, axis=(0, 1))
    centeredB = im2 - np.mean(im2, axis=(0, 1))
    sum_pix = lambda arr: np.sum(arr, axis=(0, 1))
    values = sum_pix((centeredA) * (centeredB)) / np.sqrt(
        sum_pix(np.power(centeredA, 2)) * sum_pix(np.power(centeredB, 2))
    )
    return np.mean(values)


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert rgb color to gray scales

    Parameters
    ----------
    rgb : np.ndarray
        [description]

    Returns
    -------
    np.ndarray
        dot product between two arrays for grayscale
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def get_frame(X, index, dimension, dims):
    tmp = np.reshape(X[:, index], dimension)
    return np.reshape(tmp, dims)


def video2matrix(video, k=5, scale=50):  # No Type
    """[summary]

    Parameters
    ----------
    video : [type]
        [description]
    k : int, optional
        [description], by default 5
    scale : int, optional
        [description], by default 50

    Returns
    -------
    [type]
        [description]
    """
    res = []
    for i in tqdm.tqdm(range(k * int(video.duration))):
        tmp = rgb2gray(video.get_frame(i / float(k)))
        im = Image.fromarray(tmp)
        size = tuple((np.array(im.size) * scale / 100))
        size = (int(size[0]), int(size[1]))
        new_image = np.array(im.resize(size)).astype(int)
        dimension = new_image.shape
        new = new_image.flatten()
        res.append(new)
    return np.vstack(res).T, dimension
