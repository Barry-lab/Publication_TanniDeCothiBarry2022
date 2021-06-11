
import numpy as np
from astropy.convolution import Gaussian1DKernel


def spike_counts_binned(timestamps, bin_edges, sum_window_size):
    """Returns spike counts in windows centered on each time bin.

    :param numpy.ndarray timestamps: spike timestamps
    :param numpy.ndarray bin_edges: bin edges to be used for binning spikes (should be uniform width bins).
    :param int sum_window_size: number of time bins over which to sum spikes (should be odd)
    :return: spike_counts
    :rtype: numpy.ndarray
    """
    return np.convolve(np.histogram(timestamps, bin_edges, density=False)[0],
                       np.ones(sum_window_size, dtype=np.int64), mode='same')


def convert_spike_times_to_sample_indices(timestamps, sampling_rate):
    """Assigns each spike to a sample of a signal with set sampling_rate.
    Assumes the sampled signal has same start time as spike timestamps.

    :param numpy.ndarray timestamps: shape (N,) floats listing spike times in seconds
    :param float sampling_rate: Hz
    :return:
    """
    return (timestamps * sampling_rate).astype(int)  # note this is rounding down


def filter_timestamps_to_sample_range(timestamps, sampling_rate, first_sample, last_sample):
    min_timestamp = np.array(first_sample, dtype=timestamps.dtype) / float(sampling_rate)
    max_timestamp = np.array(last_sample + 1, dtype=timestamps.dtype) / float(sampling_rate)
    return timestamps[(timestamps >= min_timestamp) & (timestamps < max_timestamp)]


def count_spikes_in_sample_bins(timestamps, sampling_rate, first_sample, last_sample, sum_samples=None,
                                sum_samples_kernel='box_car', timestamps_already_in_range=False):
    """Returns the number of timestamps in each sample bin.

    The returned array lists sample counts starting from first_sample and ending with the last_sample

    :param numpy.ndarray timestamps: timestamps of events
    :param int sampling_rate: sampling rate in Hz
    :param int first_sample: the first sample index to be included in sample_counts
    :param int last_sample: the last sample index to be included in sample_counts
    :param int sum_samples: if specified, number of each element is equal to sum of sum_samples samples
        window centered on the element (achieved with numpy.convolve using box kernel of 1s).
        If sum_samples_kernel == 'gaussian', then sum_samples specifies the standard deviation
        of the gaussian kernel in number of samples.
    :param str sum_samples_kernel: either 'box_car' (default) or 'gaussian'
        If 'gaussian', then the output is the average of samples smoothed, instead of the sum.
    :param bool timestamps_already_in_range: if False (default) timestamps are first filtered to ones
        that are in range between first_sample and last_sample. This can be turned off if input timestamps
        has already been filtered to reduce computing time.
    :return: sample_counts
    :rtype: numpy.ndarray
    """
    # Ignore all timestamps beyond the ones in range of requested samples
    if not timestamps_already_in_range:
        timestamps = filter_timestamps_to_sample_range(timestamps, sampling_rate, first_sample, last_sample)

    # Count spikes in each sample bin
    sample_counts = np.bincount(convert_spike_times_to_sample_indices(timestamps, sampling_rate) - first_sample)
    if sample_counts.size <= (last_sample - first_sample):
        sample_counts = np.concatenate((sample_counts,
                                        np.zeros((last_sample - first_sample + 1) - sample_counts.size,
                                                 dtype=sample_counts.dtype)))

    # Count sum of samples with moving window if sum_samples specified
    if not (sum_samples is None):
        if sum_samples_kernel == 'box_car':
            sample_counts = np.convolve(sample_counts, np.ones(sum_samples, dtype=np.int64), mode='same')
        elif sum_samples_kernel == 'gaussian':
            sample_counts = np.convolve(sample_counts, Gaussian1DKernel(sum_samples), mode='same')
        else:
            raise ValueError('Unexpected sum_samples_kernel value {}'.format(sum_samples_kernel))

    return sample_counts
