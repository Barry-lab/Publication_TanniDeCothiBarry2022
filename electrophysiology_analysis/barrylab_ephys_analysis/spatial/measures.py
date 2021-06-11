
import numpy as np


def ratemap_gradient(ratemap):
    """Returns the sum of gradients along first two axis of the array.

    :param numpy.ndarray ratemap: 2-D or 3-D array.
    :return: gradient
    :rtype: numpy.ndarray
    """
    gradients_y, gradients_x = np.gradient(ratemap, axis=(0, 1))
    return np.abs(gradients_x) + np.abs(gradients_y)


def ratemap_fisher_information(ratemap, min_rate=1., axis=None):
    """Returns the fisher information along first two axis of the array.

    :param numpy.ndarray ratemap: 2-D or 3-D array.
    :param float min_rate: minimum rate for fisher information to be compute at position
    :param str axis: either 'x' or 'y', specifying computing of fisher information only on
        second or first dimension or ratemap array, respectively.
    :return: fisher_information
    :rtype: numpy.ndarray
    """
    gradients_y, gradients_x = np.gradient(ratemap, axis=(0, 1))
    idx_ignore = ratemap < min_rate
    gradients_y[idx_ignore] = np.nan
    gradients_x[idx_ignore] = np.nan

    if axis is None:

        return (np.power(gradients_y, 2) / ratemap) + (np.power(gradients_x, 2) / ratemap)

    elif axis == 'x':

        return np.power(gradients_x, 2) / ratemap

    elif axis == 'y':

        return np.power(gradients_y, 2) / ratemap

    else:

        raise ValueError('Unexpected value for axis: {}'.format(axis))
