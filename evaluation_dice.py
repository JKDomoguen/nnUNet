import numpy as np

def binary_dice(s, g,):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    resize: if s and g have different shapes, resize s to match g.
    """
    assert(len(s.shape)== len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    s1 = s.sum()
    s2 = g.sum()
    dice = (2.0*s0 + 1e-5)/(s1 + s2 + 1e-5)
    return dice