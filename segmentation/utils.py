import numpy as np


def colorize_mask(mask, color):
    """
    Method used for coloring a binary mask with a given color
    Args:
        mask: 2d binary array
        color: triplet representing the color
    Return:
        colored mask
    """
    colored_mask = np.zeros(mask.shape + (3,), dtype=np.uint8)
    colored_mask[mask > 0.1] = color

    return colored_mask