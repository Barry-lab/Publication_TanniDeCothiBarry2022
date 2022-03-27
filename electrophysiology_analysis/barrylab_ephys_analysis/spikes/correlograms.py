
import numpy as np
from tqdm import tqdm
from pycorrelate.pycorrelate import pcorrelate


def create_correlation_bin_edges(max_lag, bin_size):
    """Returns bin_edges for bins that fit entirely within -max_lag and max_lag,
    with specified bin_size and symmetric around 0.

    The returned bin_edges will always have even number of elements, indicating
    odd number of bins, with the center bin centered on 0.

    :param float max_lag: maximum and minimum allowed bin edge value
    :param float bin_size: width of a bin
    :return: bin_edges
    :rtype: numpy.ndarray
    """

    edges = np.ceil(max_lag / float(bin_size)) * bin_size - bin_size / 2.

    bin_edges = np.concatenate([np.arange(-edges, edges, bin_size), np.array([edges])])

    return bin_edges[np.logical_and(bin_edges > -max_lag, bin_edges < max_lag)]


def cross_correlation(timestamps_a, timestamps_b, bin_edges, normalize=False, counts=True):
    """Returns cross-correlation of point process timestamps (e.g. spike timestamps).

    :param numpy.ndarray timestamps_a: first timestamps to cross-correlate.
    :param numpy.ndarray timestamps_b: second timestamps to cross-correlate.
    :param numpy.ndarray bin_edges: values marking correlation bin edge locations
    :param bool normalize: if True (default is False), output is normalised correlation.
    :param bool counts: if True (default), output counts per each bin.
    :return: correlation
    :rtype: numpy.ndarray
    """
    if normalize and counts:
        raise Exception('Only one of normalize or counts can be set True.')

    correlation = pcorrelate(timestamps_a, timestamps_b, bin_edges, normalize=normalize)

    if counts:
        correlation = np.int64(np.round(correlation * np.diff(bin_edges)))

    return correlation


def auto_correlation(timestamps, bin_edges, normalize=False, counts=True):
    """Returns auto-correlation of point process timestamps (e.g. spike timestamps).

    Central bin is set to zero as its real value is known.

    :param numpy.ndarray timestamps: timestamps to auto-correlate.
    :param numpy.ndarray bin_edges: values marking correlation bin edge locations
    :param bool normalize: if True (default is False), output is normalised correlation.
    :param bool counts: if True (default), output counts per each bin.
    :return: correlation
    :rtype: numpy.ndarray
    """
    correlation = cross_correlation(timestamps, timestamps, bin_edges,
                                    normalize=normalize, counts=counts)

    middle_bin = int(np.floor(len(bin_edges) / 2.0)) - 1

    correlation[middle_bin] = 0.

    return correlation


def plot_correlogram(bin_values, bin_edges, ax):
    """Plots output from :py:func:`auto_correlation` or :py:func:`cross_correlation`
    to `matplotlib` axes provided.

    :param numpy.ndarray bin_values: correlation values to plot, shape (n_bins,)
    :param numpy.ndarray bin_edges: x-axis edges for values in bin_values, shape (n_bins + 1,)
    :param ax: instance of :py:class:`matplotlib.axes.Axes`
    """

    bin_sizes = np.concatenate([bin_edges[1:] - bin_edges[:-1], np.array([0])])

    ax.bar(bin_edges, np.concatenate([bin_values, np.array([0.])]),
           width=bin_sizes, align='edge', linewidth=0)
    ax.set_xlim((bin_edges[0], bin_edges[-1]))


class CoincidentSpikeProbability(object):

    @staticmethod
    def compute_between_units_in_recording(recording, max_latency, verbose=False):
        """Computes coincident spike probability between each unit in the recording and stores results
        in unit['analysis']['coincident_spike_probability']

        :param recording: py:class:`recording_io.Recording`
        :param float max_latency: max_latency for detecting coincident firing (seconds)
        :param bool verbose: if True, progress is displayed in console (default is False)
        """

        bin_edges = create_correlation_bin_edges(
            max_latency + np.finfo(np.float64).resolution,
            max_latency * 2
        )

        unit_pairs = []
        for i in range(1, len(recording.units)):
            for j in range(i + 1, len(recording.units)):
                unit_pairs.append((i, j))

        coincident_spike_probability = np.diagflat(np.ones(len(recording.units), dtype=np.float64))
        for (i, j) in (tqdm(unit_pairs) if verbose else unit_pairs):
            coincident_spike_count = cross_correlation(recording.units[i]['timestamps'],
                                                       recording.units[j]['timestamps'],
                                                       bin_edges, normalize=False, counts=True)
            tmp = coincident_spike_count / max(len(recording.units[i]['timestamps']),
                                               len(recording.units[j]['timestamps']))
            coincident_spike_probability[i, j] = tmp
            coincident_spike_probability[j, i] = tmp

        for i, unit in enumerate(recording.units):
            unit['analysis']['coincident_spike_probability'] = coincident_spike_probability[i, :]

    @staticmethod
    def compute_between_units_in_recordings(recordings, max_latency, verbose=False):
        """Computes coincident spike probability between each unit in the recording and stores results
        in recording.analysis['coincident_spike_probability'] of the first recording as a list,
        where each element corresponds to each element in :py:attr:`Recordings.units`.

        :param recordings: py:class:`recording_io.Recordings`
        :param float max_latency: max_latency for detecting coincident firing (seconds)
        :param bool verbose: if True, progress is displayed in console (default is False)
        """

        bin_edges = create_correlation_bin_edges(
            max_latency + np.finfo(np.float64).resolution,
            max_latency * 2
        )

        unit_pairs = []
        for i in range(1, len(recordings.units)):
            for j in range(i + 1, len(recordings.units)):
                unit_pairs.append((i, j))

        coincident_spike_probability = np.diagflat(np.ones(len(recordings.units), dtype=np.float64))
        for (i, j) in (tqdm(unit_pairs) if verbose else unit_pairs):
            i_timestamps = recordings.unit_timestamps_concatenated_across_recordings(i)
            j_timestamps = recordings.unit_timestamps_concatenated_across_recordings(j)

            coincident_spike_count = cross_correlation(i_timestamps, j_timestamps,
                                                       bin_edges, normalize=False, counts=True)
            tmp = coincident_spike_count / max(len(i_timestamps), len(j_timestamps))
            coincident_spike_probability[i, j] = tmp
            coincident_spike_probability[j, i] = tmp

        recordings[0].analysis['coincident_spike_probability'] = [coincident_spike_probability[i, :]
                                                                  for i in range(len(recordings.units))]

    @staticmethod
    def violation_pairs_in_recording(recording, threshold):

        correlated_units = []
        for i, unit in enumerate(recording.units):
            correlations = unit['analysis']['coincident_spike_probability'].copy()

            correlations[i] = 0
            correlated_unit_inds = np.where(np.abs(correlations) > threshold)[0]
            if len(correlated_unit_inds) == 0:
                continue

            for i_correlated_unit in correlated_unit_inds:
                correlated_units.append(tuple(sorted([i, i_correlated_unit]) + [correlations[i_correlated_unit]]))

        # Get rid of duplicates
        correlated_units = list(set(correlated_units))

        return correlated_units

    @staticmethod
    def violation_pairs_in_recordings(recordings, threshold):

        correlated_units = []
        for i, correlations in enumerate(recordings[0].analysis['coincident_spike_probability']):
            correlations = correlations.copy()

            correlations[i] = 0
            correlated_unit_inds = np.where(np.abs(correlations) > threshold)[0]
            if len(correlated_unit_inds) == 0:
                continue

            for i_correlated_unit in correlated_unit_inds:
                correlated_units.append(tuple(sorted([i, i_correlated_unit]) + [correlations[i_correlated_unit]]))

        # Get rid of duplicates
        correlated_units = list(set(correlated_units))

        return correlated_units
