
import numpy as np
import warnings
from scipy.signal import convolve2d


def compute_xy_position_bin_inds(xy, bin_size):
    """Converts xy values to position bin indices assuming that
    the first bin edge starts at 0.

    :param xy: numpy.array position coordinate values
    :param float bin_size: size of a bin
    :return: same shape as xy, but values referring to bin indices
    :rtype: numpy.ndarray
    """
    return (xy / bin_size).astype(int)


def compute_direction_direction_bin_inds(direction, bin_size):
    """Converts direction values to direction bin indices assuming that
    direction values range from -pi to pi.

    :param numpy.ndarray direction: shape (N,) direction values in range -pi to pi
    :param float bin_size: size of a bin
    :return: same shape as xy, but values referring to bin indices
    :rtype: numpy.ndarray
    """
    return ((direction + np.pi - np.finfo(direction.dtype).resolution) / bin_size).astype(int)


def xy_spatial_window_mask(xy, xmin, xmax, ymin, ymax):
    """Returns a boolean array specifying which xy samples are within limits

    :param numpy.ndarray xy: shape (n_samples, 2)
    :param float xmin:
    :param float ymin:
    :param float xmax:
    :param float ymax:
    :return: shape (n_samples,) array with True for samples within limits
    :rtype: numpy.ndarray
    """
    xin = np.logical_and(xy[:, 0] > xmin, xy[:, 0] < xmax)
    yin = np.logical_and(xy[:, 1] > ymin, xy[:, 1] < ymax)
    return np.logical_and(xin, yin)


def convert_spike_times_to_sample_indices(timestamps, sampling_rate):
    """Assigns each spike to a sample of a signal with set sampling_rate.
    Assumes the sampled signal has same start time as spike timestamps.

    :param numpy.ndarray timestamps: shape (N,) floats listing spike times in seconds
    :param float sampling_rate: Hz
    :return:
    """
    return (timestamps * sampling_rate).astype(int)  # note this is rounding down


def circular_shift_sample_indices(indices, n_samples_shift, n_samples_total):
    """Shifts sample index values circularly by speficied amount

    :param numpy.ndarray indices: shape (N,) in dtype int
    :param int n_samples_shift: number of samples to shift, can be positive or negative
    :param int n_samples_total: total number of samples in the data
    :return: shifted indices
    :rtype: numpy.ndarray
    """
    return (indices + n_samples_shift) % n_samples_total
