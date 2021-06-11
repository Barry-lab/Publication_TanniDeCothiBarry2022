
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy import interpolate
import numpy as np
from numpy_groupies import aggregate_np as aggregate
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from tqdm import tqdm

from barrylab_ephys_analysis.external import circstats
from barrylab_ephys_analysis.blea_utils import gray_arrays_to_rgb_sequence_array, isin_along_rows_merge_with_sum
from barrylab_ephys_analysis.spatial.utils import (convert_spike_times_to_sample_indices,
                                                   compute_xy_position_bin_inds,
                                                   xy_spatial_window_mask,
                                                   circular_shift_sample_indices)
from barrylab_ephys_analysis.models.gaussian import draw_model_contours_on_image


class SpatialRatemap(object):

    """
    This class transforms position data and spike train to a spatial map of positional
    sampling and positional firing.

    Computations are done as attributes, such as :py:attr:`SpatialRatemap.spike_rates` are
    queried for the first time.

    See these methods for common use cases:
        :py:func:`SpatialRatemap.spike_rates_smoothed`
        :py:func:`SpatialRatemap.spike_rates_adaptive_smoothed`
        :py:func:`SpatialRatemap.spike_rates_shifted_smoothed`
        :py:func:`SpatialRatemap.plot`
        :py:func:`SpatialRatemap.instantiate_for_odd_even_minutes`
        :py:func:`SpatialRatemap.instantiate_for_first_last_half`
    """

    def __init__(self, xy, spiketimes, position_sampling_rate, bin_size,
                 spatial_window=None, spike_xy_ind=None, xy_mask=None, n_samples=None,
                 limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds=None):
        """Instantiates SpatialRatemap with specified parameters. Attributes like
        :py:attr:`SpatialRatemap.spike_rates` can then be queried.

        :param numpy.ndarray xy: shape (n_samples, 2) position values of x (horizontal) and y coordinates.
            Must not have NaN values, unless they are masked out using `xy_mask`.
        :param numpy.ndarray spiketimes: shape (n_spikes,) spike times of a neuron aligned to first sample of xy
        :param int position_sampling_rate: Hz
        :param float bin_size: number of spatial units (xy units) to pool into a single position bin
        :param tuple spatial_window: minimum and maximum xy values to include (x_start, x_end, y_start, y_end)
        :param spike_xy_ind: xy data indices of each spike. This is computed if not provided
        :param numpy.ndarray xy_mask: boolean array specifying which samples of xy are used
            for computing the spatial maps, with True values for position samples to be included.
        :param int n_samples: if provided, the fixed number of samples are used to compute ratemaps
            at each spatial position. For position bins where there are fewer than `n_samples` samples,
            the no samples are used, resulting in `numpy.nan` values for those bins. For position bins
            where more than `n_samples` samples are available, `n_samples` are selected randomly.
            This is applied after any masking with `spatial_window` and/or `xy_mask`.
        :param numpy.ndarray limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds:
            shape (n_samples, 2) final position bin indices from a previous SpatialRatemap instance
            as returned by SpatialRatemap.xy_shifted_masked_position_bin_inds. If provided, this is used
            to restrict the sampled bins to those present in the provided array. The number of samples
            used is also restricted (sub-sampled) to the number of samples in the provided array.
            This sub-sampling is done across the recording not on a bin-by-bin basis.
            If there are fewer samples available than in the provided array, an Exception is raised.
        """

        self.check_if_xy_negative_values_included(xy, spatial_window, xy_mask)

        self._xy = xy
        self._spiketimes = spiketimes
        self._position_sampling_rate = float(position_sampling_rate)
        self._bin_size = float(bin_size)
        self._spatial_window = spatial_window
        self._spike_xy_ind = spike_xy_ind
        self._input_xy_mask = xy_mask
        self._n_samples = n_samples
        self._previous_xy_shifted_masked_position_bin_inds = \
            limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds

        self._xy_mask = None
        self._xy_shifted_position_bin_inds = None
        self._dwell_counts = None
        self._xy_in_sampled_bins = None
        self._spike_counts = None
        self._spike_rates = None

    @staticmethod
    def check_if_xy_negative_values_included(xy, spatial_window, xy_mask):

        if not np.any(xy[~np.isnan(xy)].flatten() < 0):
            return

        if not (spatial_window is None) and not np.any(np.array(spatial_window) < 0):
            return

        if not (xy_mask is None) and not np.any(xy[xy_mask, :].flatten() < 0):
            return

        raise ValueError('If xy contains negative values, these must be excluded based on\n'
                         + 'spatial_window or xy_mask or both')

    @staticmethod
    def shift_xy(xy, spatial_window):
        xy = xy.copy()
        xy[:, 0] -= spatial_window[0]
        xy[:, 1] -= spatial_window[2]
        return xy

    @staticmethod
    def compute_spatial_window(xy):
        return (np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0]),
                np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1]))

    @property
    def spatial_window(self):
        """Returns minimum and maximum xy values included in ratemap

        :return: spatial_window (x_start, x_end, y_start, y_end) position data window
                 If not provided will be min and max of values in xy
        :rtype: tuple
        """
        if self._spatial_window is None:
            self._spatial_window = SpatialRatemap.compute_spatial_window(self._xy)
        return deepcopy(self._spatial_window)

    @staticmethod
    def compute_bin_of_xy_after_shifting(xy, spatial_window, binsize):
        """Returns the ratemap bin indices corresponding to a single sample or set of xy values.

        :param numpy_ndarray xy: shape (n_samples, 2), where columns are x and y position values.
        :param tuple spatial_window: spatial window used to compute ratemap (x_start, x_end, y_start, y_end)
            x_start and y_start values are subtracted from respective columns of `xy`.
        :param float binsize: bin size used to make ratemaps
        :return: position indices, same shape as xy
        :rtype: numpy.ndarray
        """
        return compute_xy_position_bin_inds(SpatialRatemap.shift_xy(xy, spatial_window), binsize)

    def create_new_mask_for_subset_based_on_n_samples(self, xy, n_samples, xy_mask=None):
        """Returns xy_mask for samples in xy such that exactly n_samples are included for each 2D position.

        If a given position has more samples, then random subset is selected. If fewer samples,
        then all samples for that position are excluded.

        :param numpy.ndarray xy: shape (n_samples, 2) position values of x (horizontal) and y coordinates.
            Must not have NaN values, unless they are masked out using `xy_mask`.
        :param int n_samples: the fixed number of samples for each spatial position bin
        :param numpy.ndarray xy_mask: boolean array specifying which samples of `xy` to include for before
            ensuring equal sampling with True values for `xy` samples to be included.
            If not provided, it is assumed all samples are to be included.
        :return: xy_mask shape (n_samples, 2)
        :rtype: numpy.ndarray
        """
        # Create fully True xy_mask if not provided
        if xy_mask is None:
            xy_mask = np.ones(xy.shape[0], dtype=np.bool)

        # Pre-compute True value indices for later indexing into xy_mask
        xy_mask_true_indices = np.where(xy_mask)[0]

        # Compute position bin indices for positions in the xy_mask
        xy_position_bin_inds = \
            self.compute_bin_of_xy_after_shifting(xy[xy_mask, :], self.spatial_window, self._bin_size)

        # Iterate over all 2D bins present in xy_position_bin_inds
        x_start, y_start = np.min(xy_position_bin_inds, axis=0)
        x_end, y_end = np.max(xy_position_bin_inds, axis=0)
        for x_bin in range(x_start, x_end + 1):
            for y_bin in range(y_start, y_end + 1):

                # Find indices in xy_position_bin_inds that match the given x_bin and y_bin
                indices = np.where((xy_position_bin_inds[:, 0] == x_bin) & (xy_position_bin_inds[:, 1] == y_bin))[0]

                # Skip next steps if no indices are found in xy_position_bin_inds to match these x_bin and y_bin
                if len(indices) == 0:
                    continue

                # If not enough samples are present, set all to False.
                # If too many samples are present, select a random n_samples to leave True and set others to False
                # Otherwise do nothing and continue with the loop as the number of samples matches perfectly
                if len(indices) < n_samples:
                    xy_mask[xy_mask_true_indices[indices]] = False
                elif len(indices) > n_samples:
                    np.random.shuffle(indices)
                    xy_mask[xy_mask_true_indices[indices[n_samples:]]] = False
                else:
                    continue

        return xy_mask

    @staticmethod
    def limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds(
            xy_shifted_position_bin_inds, previous_xy_shifted_masked_position_bin_inds, xy_mask
    ):
        """Limits the xy_mask to previously specified position bin indices and also the same
        total number of samples.
        """
        n_samples_to_include = previous_xy_shifted_masked_position_bin_inds.shape[0]
        xy_mask_inds = np.where(xy_mask)[0]

        xy_shifted_position_bin_inds_masked = xy_shifted_position_bin_inds[xy_mask_inds, :]
        unique_previous_xy_shifted_masked_position_bin_inds = np.unique(previous_xy_shifted_masked_position_bin_inds,
                                                                        axis=0)
        xy_mask_inds_include_idx = isin_along_rows_merge_with_sum(xy_shifted_position_bin_inds_masked,
                                                                  unique_previous_xy_shifted_masked_position_bin_inds)

        xy_mask_inds_include = xy_mask_inds[xy_mask_inds_include_idx]

        if xy_mask_inds_include.size < n_samples_to_include:
            raise Exception('Available samples are fewer than those in previous_xy_shifted_masked_position_bin_inds\n'
                            +'Unable to subsample')
        elif xy_mask_inds_include.size > n_samples_to_include:
            xy_mask_inds_include = \
                xy_mask_inds_include[np.random.permutation(xy_mask_inds_include.size)[:n_samples_to_include]]
        else:
            pass

        xy_mask = np.zeros(xy_mask.shape, dtype=np.bool)
        xy_mask[xy_mask_inds_include] = True

        return xy_mask

    @property
    def xy_mask(self):
        """Boolean array specifying which samples of :py:attr:`SpatialRatemap.xy` are used
        for computing the spatial maps
        """
        if self._xy_mask is None:
            if self._input_xy_mask is None:
                xy_mask = xy_spatial_window_mask(self.shift_xy(self._xy, self.spatial_window), *self.spatial_window)
            else:
                xy_mask = self._input_xy_mask & xy_spatial_window_mask(self.shift_xy(self._xy, self.spatial_window),
                                                                       *self.spatial_window)
            if self._n_samples:
                xy_mask = self.create_new_mask_for_subset_based_on_n_samples(
                    self._xy.copy(), self._n_samples, xy_mask=xy_mask
                )
            if not (self._previous_xy_shifted_masked_position_bin_inds is None):
                xy_mask = self.limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds(
                    self.xy_shifted_position_bin_inds, self._previous_xy_shifted_masked_position_bin_inds, xy_mask
                )

            self._xy_mask = xy_mask

        return self._xy_mask.copy()

    @property
    def xy_shifted_and_masked(self):
        """:py:attr:`SpatialRatemap._xy` shifted by
        :py:attr:`SpatialRatemap.spatial_window` minimum values.
        The minimum values are subtracted from respective columns.
        Subset of the resultinig array is given based on :py:attr:`SpatialRatemap.xy_mask`

        :rtype: numpy.ndarray
        """
        return self.shift_xy(self._xy[self.xy_mask, :], self.spatial_window)

    @property
    def x_position_bin_range(self):
        return self.spatial_window[1] - self.spatial_window[0]

    @property
    def num_x_position_bins(self):
        return int((self.x_position_bin_range // self._bin_size) + 1)

    @property
    def y_position_bin_range(self):
        return self.spatial_window[3] - self.spatial_window[2]

    @property
    def num_y_position_bins(self):
        return int((self.y_position_bin_range // self._bin_size) + 1)

    @property
    def position_bins(self):
        """(x_position_bin_centers, y_position_bin_centers)
        """
        return (np.arange(0, self.x_position_bin_range, self._bin_size) + self._bin_size / 2.,
                np.arange(0, self.y_position_bin_range, self._bin_size) + self._bin_size / 2.)

    @property
    def xy_shifted_position_bin_inds(self):
        """Position bin indices that each xy value belongs to after shifting to spatial window.
        """
        if self._xy_shifted_position_bin_inds is None:
            self._xy_shifted_position_bin_inds = \
                self.compute_bin_of_xy_after_shifting(self._xy, self.spatial_window, self._bin_size)
        return self._xy_shifted_position_bin_inds.copy()

    @property
    def xy_shifted_masked_position_bin_inds(self):
        """Position bin indices that each xy value belongs to after shifting to spatial window and masking.
        """
        return self.xy_shifted_position_bin_inds[self.xy_mask, :]

    @property
    def dwell_counts(self):
        if self._dwell_counts is None:
            self._dwell_counts = aggregate(
                self.xy_shifted_masked_position_bin_inds.T, 1.,
                size=(self.num_x_position_bins, self.num_y_position_bins)
            ).T
        return self._dwell_counts.copy()

    @property
    def sampled_position_bins(self):
        return self.dwell_counts > 0

    @property
    def xy_in_sampled_bins(self):
        """Boolean mask for xy where values are True for samples in bins sampled for ratemap.
        """
        if self._xy_in_sampled_bins is None:

            xy_bin_inds = self.compute_bin_of_xy_after_shifting(self._xy, self.spatial_window, self._bin_size)
            xy_bin_inds_positive = np.where(np.all(xy_bin_inds >= 0, axis=1))[0]

            idx_xy_bin_inds_positive_sampled = isin_along_rows_merge_with_sum(
                xy_bin_inds[xy_bin_inds_positive],
                np.fliplr(np.stack(np.where(self.sampled_position_bins), axis=1)),
            )

            self._xy_in_sampled_bins = np.zeros(self._xy.shape[0], dtype=np.bool)
            self._xy_in_sampled_bins[xy_bin_inds_positive[idx_xy_bin_inds_positive_sampled]] = True

        return self._xy_in_sampled_bins

    @property
    def dwell_time(self):
        return self.dwell_counts / self._position_sampling_rate

    @property
    def spike_xy_ind(self):
        """xy data indices of each spike.
        """
        if self._spike_xy_ind is None:
            self._spike_xy_ind = convert_spike_times_to_sample_indices(
                self._spiketimes, self._position_sampling_rate)
        return self._spike_xy_ind.copy()

    @staticmethod
    def apply_xy_mask_to_spike_xy_ind(spike_xy_ind, xy_mask):
        return spike_xy_ind[np.isin(spike_xy_ind, np.where(xy_mask)[0])]

    @property
    def spike_xy_ind_masked(self):
        """:py:attr:`SpatialRatemap.spike_xy_ind` only including the elements that refer to
        :py:attr:`SpatialRatemap.xy` indices included in :py:attr:`SpatialRatemap.xy_masked`
        """
        return SpatialRatemap.apply_xy_mask_to_spike_xy_ind(self.spike_xy_ind, self.xy_mask)

    @staticmethod
    def compute_spike_counts(xy_position_bin_inds, spike_xy_ind_masked,
                             n_x_bins, n_y_bins):
        """Returns count of spikes in each 2D position bin.

        :param numpy.ndarray xy_position_bin_inds: shape (n_samples, 2) position bin indices that each xy value
            belongs to after shifting to spatial window, but before application of :py:attr:`SpatialRatemap.xy_mask`.
        :param numpy.ndarray spike_xy_ind_masked: shape (n_spikes,) indices of `xy_position_bin_inds` of each
            spike, filtered for spikes that refer to samples included based on :py:attr:`SpatialRatemap.xy_mask`.
        :param int n_x_bins: number of position bins in x-axis (along 2nd dimension of output array)
        :param int n_y_bins: number of position bins in y-axis (along 1st dimension of output array)
        :return: spike_counts shape (n_y_bins, n_x_bins)
        :rtype: numpy.ndarray
        """
        return aggregate(xy_position_bin_inds[spike_xy_ind_masked, :].T, 1.,
                         size=(n_x_bins, n_y_bins)).T

    @property
    def spike_counts(self):
        if self._spike_counts is None:
            self._spike_counts = SpatialRatemap.compute_spike_counts(
                self.xy_shifted_position_bin_inds, self.spike_xy_ind_masked,
                self.num_x_position_bins, self.num_y_position_bins
            )
        return self._spike_counts.copy()

    @staticmethod
    def compute_ratemap(spike_counts, dwell_time, sampled_position_bins):
        """Returns ratemap computed based on spike counts and dwell time in each position bin.

        :param numpy.ndarray spike_counts: shape (n_y_bins, n_x_bins) count of spikes in each position bin
        :param numpy.ndarray dwell_time: shape (n_y_bins, n_x_bins) total duration of samples in each position bin (s)
        :param numpy.ndarray sampled_position_bins: shape (n_y_bins, n_x_bins) boolean array specifying which
            position bins were not sampled. These are set to np.nan
        :return: spike_rates shape (n_y_bins, n_x_bins)
        :rtype: numpy.ndarray
        """
        spike_rates = spike_counts.copy()
        spike_rates[sampled_position_bins] = \
            spike_counts[sampled_position_bins] / dwell_time[sampled_position_bins]
        spike_rates[~sampled_position_bins] = np.nan

        return spike_rates

    @property
    def spike_rates(self):
        if self._spike_rates is None:
            self._spike_rates = SpatialRatemap.compute_ratemap(
                self.spike_counts, self.dwell_time, self.sampled_position_bins)
        return self._spike_rates.copy()

    @staticmethod
    def smooth(data, ignore_idx, n_bins=2, method='gaussian'):
        """Applies smoothing to data using normalized convolution to deal with np.nan values.

        :param numpy.ndarray data: shape (num_y_bins, num_x_bins), dtype int
        :param numpy.ndarray ignore_idx: shape (num_y_bins, num_x_bins), dtype bool
        :param float n_bins: size of the smoothing kernel in bins
        :param method: 'boxcar' or 'gaussian'
        :return: smoothed_data
        :rtype: numpy.ndarray
        """

        if method == 'boxcar':
            filter_func = uniform_filter
        elif method == 'gaussian':
            filter_func = gaussian_filter
        else:
            raise ValueError('Unknown smoothing method {}'.format(method))

        # Use normalized convolution to compute smoothed data
        # as in this example https://stackoverflow.com/a/36307291
        data_arr = data.copy()
        data_arr[np.isnan(data)] = 0
        data_arr_smoothed = filter_func(data_arr, n_bins, mode='constant', cval=0)
        nan_array = np.ones(data.shape, dtype=data.dtype)
        nan_array[np.isnan(data)] = 0
        nan_array_smoothed = filter_func(nan_array, n_bins, mode='constant', cval=0)

        # Create smooth data array
        smoothed_data = np.zeros(data.shape,
                                 dtype=data_arr.dtype)
        # Compute smoothed data at positions where it was above 0 in nan_array
        idx_above_zero = nan_array_smoothed > 0
        smoothed_data[idx_above_zero] = \
            data_arr_smoothed[idx_above_zero] / nan_array_smoothed[idx_above_zero]
        # Set bins that are to be ignored back to np.nan
        smoothed_data[ignore_idx] = np.nan

        return smoothed_data

    def spike_rates_smoothed(self, n_bins=2, method='gaussian', ignore_0_sample_bins=True,
                             min_dwell_time=None):
        """Returns spike rates computed after smoothing :py:attr:`SpatialRatemap.spike_counts`
        and :py:attr:`SpatialRatemap.dwell_time`.

        Smoothing is performed such that bins outside the environment do not contribute to
        output (i.e. effectively the kernel crop method is used).
        This same method is applied to 0 sample bins, if `ignore_0_sample_bins=True` (default).

        :param float n_bins: size of the smoothing kernel in bins
        :param str method: 'boxcar' or 'gaussian'
        :param bool ignore_0_sample_bins: if True (default), bins with 0 samples are ignored
            in smoothing and set to `numpy.nan` in the output.
        :param float min_dwell_time: position bins with dwell time below this value (in seconds)
            are handled as zero sample bins.
        :return: smoothed_spike_rates
        :rtype: numpy.ndarray
        """

        if method == 'boxcar':
            filter_func = uniform_filter
        elif method == 'gaussian':
            filter_func = gaussian_filter
        else:
            raise ValueError('Unknown smoothing method {}'.format(method))

        spike_counts = self.spike_counts
        dwell_time = self.dwell_time

        include_bins = np.ones(spike_counts.shape, dtype=np.bool)

        if ignore_0_sample_bins:
            include_bins = include_bins & self.sampled_position_bins

        if not (min_dwell_time is None):
            include_bins = include_bins & (dwell_time < min_dwell_time)

        spike_counts[~include_bins] = 0
        dwell_time[~include_bins] = 0

        spike_counts = filter_func(spike_counts, n_bins, mode='constant', cval=0)
        dwell_time = filter_func(dwell_time, n_bins, mode='constant', cval=0)

        spike_rates = np.zeros(spike_counts.shape, dtype=dwell_time.dtype) * np.nan
        spike_rates[include_bins] = spike_counts[include_bins] / dwell_time[include_bins]

        return spike_rates

    def spike_rates_shifted(self, shifted_seconds):
        """Returns same as :py:attr:`SpatialRatemap.spike_rates` but computed with
        circularly shifted spike timestamps.

        :param float shifted_seconds: spike train shift in time. Can be positive or negative.
        :return: spike_rates, shape (num_y_position_bins, num_x_position_bins)
        :rtype: numpy.ndarray
        """
        n_shifted_xy_bins = int(np.round(shifted_seconds * self._position_sampling_rate))

        spike_xy_ind = np.apply_along_axis(circular_shift_sample_indices, 0,
                                           self.spike_xy_ind, n_shifted_xy_bins, self._xy.shape[0])
        spike_xy_ind_masked = SpatialRatemap.apply_xy_mask_to_spike_xy_ind(spike_xy_ind, self.xy_mask)

        spike_counts = SpatialRatemap.compute_spike_counts(
            self.xy_shifted_position_bin_inds, spike_xy_ind_masked,
            self.num_x_position_bins, self.num_y_position_bins
        )

        return SpatialRatemap.compute_ratemap(spike_counts, self.dwell_time,
                                              self.sampled_position_bins)

    def spike_rates_shifted_smoothed(self, shifted_seconds, n_bins=2, method='gaussian'):
        """Same as :py:func:`SpatialRatemap.spike_rates_shifted` but with smoothing applied
        using :py:func:`SpatialRatemap.smooth`
        """
        return SpatialRatemap.smooth(self.spike_rates_shifted(shifted_seconds),
                                     ~self.sampled_position_bins,
                                     n_bins=n_bins, method=method)

    def compute_mean_speed_per_bin(self, speed):
        """Returns an array the same shape as :py:attr:`SpatialRatemap.spike_rates` where
        each value is the mean speed in that bin.

        Mean speed is computed while ignoring numpy.nan values in input array. Any unsampled
        bins are numpy.nan

        :param numpy.ndarray speed: shape (n_samples,) where n_samples is the same as
            in the xy parameter input to this class instance during instantiation.
        :return: speed_map shape  (num_y_position_bins, num_x_position_bins)
        :rtype: numpy.ndarray
        """
        return aggregate(
            self.xy_shifted_masked_position_bin_inds.T, speed[self.xy_mask],
            func='nanmean', size=(self.num_x_position_bins, self.num_y_position_bins),
            fill_value=np.nan
        ).T

    @staticmethod
    def compute_bin_centers_from_spatial_window_and_shape(data_shape, spatial_window):
        """Returns the xy position bin values corresponding to bin indices in the associated array.

        :param tuple data_shape: (n_y_bins, n_x_bins)
        :param tuple spatial_window: minimum and maxium xy values to include (x_start, x_end, y_start, y_end)
        :return: x_bin_centers, y_bin_centers
        """
        x_bin_step = (spatial_window[1] - spatial_window[0]) / float(data_shape[1])
        x_bin_centers = np.arange(spatial_window[0], spatial_window[1], x_bin_step) + (x_bin_step / 2.)

        y_bin_step = (spatial_window[3] - spatial_window[2]) / float(data_shape[0])
        y_bin_centers = np.arange(spatial_window[2], spatial_window[3], y_bin_step) + (y_bin_step / 2.)

        return x_bin_centers, y_bin_centers

    @staticmethod
    def compute_bin_edges_from_spatial_window_and_shape(data_shape, spatial_window, bin_size):
        """Returns the xy position bin values corresponding to edes of bin indices in the associated array.

        :param tuple data_shape: (n_y_bins, n_x_bins)
        :param tuple spatial_window: minimum and maxium xy values to include (x_start, x_end, y_start, y_end)
        :return: x_bin_edges, y_bin_edges
        """
        x_bin_centers, y_bin_centers = \
            SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(data_shape, spatial_window)
        x_bin_edges = np.concatenate((x_bin_centers - bin_size / 2., [x_bin_centers[-1] + bin_size / 2.]))
        y_bin_edges = np.concatenate((y_bin_centers - bin_size / 2., [y_bin_centers[-1] + bin_size / 2.]))
        return x_bin_edges, y_bin_edges

    @staticmethod
    def compute_ratemap_shape(spatial_window, binsize):
        """Returns the shape of the ratemap generated with specific spatial_window and binsize settings

        :param spatial_window:
        :param binsize:
        :return: shape
        :rtype: tuple
        """
        return tuple(map(int,
            SpatialRatemap.compute_bin_of_xy_after_shifting(
                np.array([[spatial_window[3] + binsize,
                           spatial_window[1] + binsize]],
                         dtype=np.float32) - np.finfo(np.float32).resolution,
                spatial_window, binsize
            ).squeeze()
        ))

    @staticmethod
    def plot_contours(contour, ax, linewidth=3, color=None):
        image_shape = ax.get_images()[0].properties()['array'].shape
        spatial_window = ax.get_images()[0].properties()['extent']

        x_bins, y_bins = SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(
            image_shape, spatial_window
        )

        contour = np.concatenate(contour, axis=0)
        contour = np.append(contour, contour[0:1, :], axis=0)

        ax.plot(x_bins[contour[:, 0]], y_bins[contour[:, 1]], linewidth=linewidth, color=color)

    @staticmethod
    def plot(data, spatial_window, ax=None, colorbar=False, show=False, colorbar_size=5.,
             contours=None, fitted_models=None, ellipse_params=None, **kwargs):
        """Can be used to plot ratemaps using :py:func:`matplotlib.pyplot.imshow`

        :param numpy.ndarray data: shape (y_bins, x_bins). np.nan is left blank
        :param spatial_window: :py:attr:`SpatialRatemap.spatial_window`
        :param matplotlib.axes._subplots.AxesSubplot ax: if provided, `ax.imshow` is used instead.
        :param bool colorbar: if True, colorbar axes is appended to image axes
        :param bool show: if True, :py:func:`matplotlib.pyplot.show` is called in the end.
        :param float colorbar_size: specifies colorbar width as a percentage of image axes width
        :param list contours: contours output from :py:func:`cv2.findContours`
        :param list fitted_models: :py:func:`barrylab_ephys_analysis.models.gaussian.fit_2d_gaussians_to_2d_data` output
        :param kwargs: passed onto `imshow` method as `**kwargs`.
        :return: None
        """

        if ax is None:
            imshow = plt.imshow
        else:
            imshow = ax.imshow

        axes_image = imshow(
            data, extent=(spatial_window[0], spatial_window[1],
                          spatial_window[3], spatial_window[2]),
            **kwargs
        )

        if ax is None:
            ax = plt.gca()

        if not (contours is None):
            for contour in contours:
                SpatialRatemap.plot_contours(contour, ax)

        if not (fitted_models is None):
            for fitted_model in fitted_models:
                draw_model_contours_on_image(fitted_model, ax, 3, linewidths=[0, 1, 1, 1],
                                             linestyles='dashed', colors='black')

        if not (ellipse_params is None):
            for ellipse_param in ellipse_params:
                ellipse = patches.Ellipse(
                    (ellipse_param['centroid_x'], ellipse_param['centroid_y']),
                    ellipse_param['major_axis'], ellipse_param['minor_axis'],
                    ellipse_param['orientation'] / np.pi * 180,
                    fill=False, linewidth=4, linestyle=':', color='black'
                )
                ax.add_patch(ellipse)

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='{}%'.format(colorbar_size), pad=0.05)
            ax.figure.colorbar(axes_image, cax=cax, ax=ax)

        if show:
            plt.show()

    @staticmethod
    def plot_posterior_sequence(posteriors, spatial_window, ax=None, color_start=(1, 0, 0), color_end=(0, 0, 1),
                                legend_kwargs=None, additional_legend_handles=None):
        """Can be used to plot ratemaps using :py:func:`matplotlib.pyplot.imshow`

        :param list posteriors: list of numpy.ndarray with shape (y_bins, x_bins)
        :param spatial_window: :py:attr:`SpatialRatemap.spatial_window`
        :param matplotlib.axes._subplots.AxesSubplot ax: if provided, `ax.imshow` is used instead.
        :param tuple color_start: (R, G, B) mapping of first array in `posteriors`
        :param tuple color_end: (R, G, B) mapping of last array in `posteriors`
        :param dict legend_kwargs: if provided, legend will be drawn with `ax.legend`. Example kwargs:
            {'bbox_to_anchor': (1.02, 1), 'loc': 'upper left'}
        :return: None
        """

        if ax is None:
            imshow = plt.imshow
        else:
            imshow = ax.imshow

        imshow(gray_arrays_to_rgb_sequence_array(posteriors, color_start, color_end),
               extent=(spatial_window[0], spatial_window[1], spatial_window[3], spatial_window[2]))

        if not (legend_kwargs is None):

            if ax is None:
                ax = plt.gca()

            if additional_legend_handles is None:
                additional_legend_handles = []

            ax.legend(handles=([patches.Patch(facecolor=color_start, edgecolor=color_start, label='start'),
                                patches.Patch(facecolor=color_end, edgecolor=color_end, label='end')]
                               + additional_legend_handles),
                      **legend_kwargs)

    @classmethod
    def instantiate_for_odd_even_minutes(cls, *args, **kwargs):
        """Returns two instances :py:class:`SpatialRatemap` with different `xy_mask`,
        one with odd and the other with even minutes included.

        Input arguments are exactly the same as for :py:class:`SpatialRatemap`
        """

        # Extract variables from input arguments
        position_sampling_rate = args[2]
        if not ('xy_mask' in kwargs):
            xy = args[0]
            xy_mask = np.ones(xy.shape[0], dtype=np.bool)
        else:
            xy_mask = kwargs['xy_mask'].copy()

        # Create separate kwargs dicts for odd and even minutes
        kwargs_odd = deepcopy(kwargs)
        kwargs_odd['xy_mask'] = xy_mask.copy()
        kwargs_even = deepcopy(kwargs)
        kwargs_even['xy_mask'] = xy_mask.copy()

        # Find odd and even minutes of the recording
        timestep = 1 / float(position_sampling_rate)
        xy_minutes = np.arange(0, (timestep - np.finfo(type(timestep)).eps) * kwargs['xy_mask'].size, timestep)
        xy_minutes = np.int64(np.ceil(xy_minutes))
        xy_odd = np.mod(xy_minutes, 2) == 1
        xy_even = np.mod(xy_minutes, 2) == 0

        # Edit xy_mask for odd and even minutes
        kwargs_odd['xy_mask'] = np.logical_and(kwargs_odd['xy_mask'], xy_odd)
        kwargs_even['xy_mask'] = np.logical_and(kwargs_even['xy_mask'], xy_even)

        return cls(*args, **kwargs_odd), cls(*args, **kwargs_even)

    @classmethod
    def instantiate_for_first_last_half(cls, *args, **kwargs):
        """Returns two instances :py:class:`SpatialRatemap` with different `xy_mask`,
        one with first half and the other with second half of the recording included.

        Input arguments are exactly the same as for :py:class:`SpatialRatemap`
        """

        # Extract variables from input arguments
        if not ('xy_mask' in kwargs):
            xy = args[0]
            xy_mask = np.ones(xy.shape[0], dtype=np.bool)
        else:
            xy_mask = kwargs['xy_mask'].copy()

        # Create separate kwargs dicts for first and second half
        kwargs_first = deepcopy(kwargs)
        kwargs_first['xy_mask'] = xy_mask.copy()
        kwargs_second = deepcopy(kwargs)
        kwargs_second['xy_mask'] = xy_mask.copy()

        # Edit xy_mask for first and second halves
        kwargs_first['xy_mask'][int(kwargs_first['xy_mask'].size * 0.5):] = False
        kwargs_second['xy_mask'][:int(kwargs_second['xy_mask'].size * 0.5)] = False

        return cls(*args, **kwargs_first), cls(*args, **kwargs_second)

    @staticmethod
    def compute_ratemap_nan_adjacent_non_nans(ratemap):
        """Returns count of adjacent non-NaN values for each NaN value in input.

        Input elements that are not NaN are set to NaN in the output.

        :param numpy.ndarray ratemap: shape (N, M)
        :return: non_nan_count
        :rtype: numpy.ndarray
        """

        nan_binary = np.isnan(ratemap).astype(np.float16)

        nan_binary_diff_d0 = np.diff(nan_binary, axis=0)
        nan_binary_diff_d0a = nan_binary_diff_d0.copy()
        nan_binary_diff_d0a[nan_binary_diff_d0a < 0] = 0
        nan_binary_diff_d0b = -nan_binary_diff_d0.copy()
        nan_binary_diff_d0b[nan_binary_diff_d0b < 0] = 0

        nan_binary_diff_d1 = np.diff(nan_binary, axis=1)
        nan_binary_diff_d1a = nan_binary_diff_d1.copy()
        nan_binary_diff_d1a[nan_binary_diff_d1a < 0] = 0
        nan_binary_diff_d1b = -nan_binary_diff_d1.copy()
        nan_binary_diff_d1b[nan_binary_diff_d1b < 0] = 0

        non_nan_count_d0 = (
            np.concatenate((np.zeros((1, nan_binary.shape[1])), nan_binary_diff_d0a), axis=0)
            + np.concatenate((nan_binary_diff_d0b, np.zeros((1, nan_binary.shape[1]))), axis=0)
        )
        non_nan_count_d0[~np.isnan(ratemap)] = np.nan

        non_nan_count_d1 = (
            np.concatenate((np.zeros((nan_binary.shape[0], 1)), nan_binary_diff_d1a), axis=1)
            + np.concatenate((nan_binary_diff_d1b, np.zeros((nan_binary.shape[0], 1))), axis=1)
        )
        non_nan_count_d1[~np.isnan(ratemap)] = np.nan

        return non_nan_count_d0 + non_nan_count_d1

    @staticmethod
    def interpolate_nans_in_ratemap(ratemap, method='linear', min_non_nan_count=None):
        """Returns ratemap with NaN values interpolated where possible based on criteria.

        NaN values remain in groups adjacent to array edges on 2 or more sides.

        :param numpy.ndarray ratemap: shape (N, M)
        :param str method: interpolation method, e.g. 'linear' or 'cubic'.
            See :py:func:`scipy.interpolate.griddata` for more information.
        :param int min_non_nan_count: if provided, values are set to back to NaN after interpolation
            where there are fewer than `min_non_nan_count` adjacent non-NaN values to original NaN values.
        :return: interpolated_ratemap
        :rtype: numpy.ndarray
        """
        xx, yy = np.meshgrid(np.arange(ratemap.shape[1]), np.arange(ratemap.shape[0]))
        interpolated_ratemap = interpolate.griddata(
            (yy[~np.isnan(ratemap)], xx[~np.isnan(ratemap)]), ratemap[~np.isnan(ratemap)],
            (yy, xx), method=method
        )

        if not (min_non_nan_count is None):
            adjacent_non_nan_count = SpatialRatemap.compute_ratemap_nan_adjacent_non_nans(ratemap)
            adjacent_non_nan_count[np.isnan(adjacent_non_nan_count)] = min_non_nan_count + 1
            idx = adjacent_non_nan_count < min_non_nan_count
            interpolated_ratemap[idx] = np.nan

        return interpolated_ratemap

    @staticmethod
    def compute_direction_filtered_ratemap(direction_bin_center, direction_bin_width, direction, xy,
                                           timestamps, bin_size, position_sampling_rate, spatial_window, xy_mask,
                                           n_smoothing_bins=None, smoothing_method=None, interpolate_nans=False,
                                           min_non_nan_count=None):
        """Returns a list of direction filtered ratemaps for requested direction.

        :param float direction_bin_center: directional filtering window centers same units as `direction`
        :param float direction_bin_width: width of directional window in same units as `direction`.
        :param numpy.ndarray direction: shape (n_samples,) array of direction values (-pi to pi) for each `xy` value
        :param numpy.ndarray xy: shape (n_samples, 2) regularly sampled position data
            columns: (x values - horizontal axis, y values - vertical axis)
        :param numpy.ndarray timestamps: shape (n_spikes,) timestamps of unit spikes
        :param float bin_size: spatial bin width for binning xy
        :param int position_sampling_rate: sampling rate of position data
        :param tuple spatial_window: minimum and maximum xy values to include (x_start, x_end, y_start, y_end)
        :param numpy.ndarray xy_mask: boolean array specifying which samples of xy are used
            for computing the spatial maps, with True values for position samples to be included.
        :param n_smoothing_bins: number of spatial bins to use for smoothing (sigma for gaussian method)
        :param str smoothing_method: 'gaussian' or 'boxcar
        :param bool interpolate_nans: if True, NaN values are interpolated before smoothing. Default is False.
        :param int min_non_nan_count: if provided, values are set to back to NaN after interpolation
            where there are fewer than `min_non_nan_count` adjacent non-NaN values to original NaN values.
        :return: direction_filtered_ratemap
        :rtype: numpy.ndarray
        """
        max_allowed_deviation_from_target_angle = direction_bin_width / 2.
        deviation_from_target_angle = np.abs(circstats.difference(direction, direction_bin_center))
        deviation_from_target_angle[np.isnan(deviation_from_target_angle)] = \
            max_allowed_deviation_from_target_angle + 1
        directional_mask = deviation_from_target_angle < max_allowed_deviation_from_target_angle
        ratemap = SpatialRatemap(xy, timestamps, position_sampling_rate, bin_size,
                                 spatial_window, xy_mask=(xy_mask & directional_mask))

        spike_rates = ratemap.spike_rates.copy()

        if interpolate_nans:

            spike_rates = SpatialRatemap.interpolate_nans_in_ratemap(spike_rates,
                                                                     min_non_nan_count=min_non_nan_count)

        if not (n_smoothing_bins is None) and not (smoothing_method is None):

            spike_rates = SpatialRatemap.smooth(spike_rates, np.isnan(spike_rates),
                                                n_bins=n_smoothing_bins, method=smoothing_method)

        return spike_rates

    @staticmethod
    def compute_direction_filtered_ratemaps(direction_bin_centers, direction_bin_width, **kwargs):
        """Returns a list of direction filtered ratemaps for each requested direction.

        :param list direction_bin_centers: directional filtering window centers
        :param float direction_bin_width: width of directional window in same units as `direction`.
        :param kwargs: additional keyword arguments as required by
            :py:func:`SpatialRatemap.compute_direction_filtered_ratemap`
        :return: ratemaps_list where elements correspond to those in `direction_bin_centers`
        :rtype: list
        """
        return [SpatialRatemap.compute_direction_filtered_ratemap(direction_bin_center, direction_bin_width,
                                                                  **kwargs, interpolate_nans=False)
                for direction_bin_center in direction_bin_centers]

    @staticmethod
    def compute_all_ratemaps_for_unit(unit, xy, bin_size, n_smoothing_bins,
                                      smoothing_method, adaptive_smoothing_alpha,
                                      position_sampling_rate, xy_mask=None,
                                      direction_filter_kwargs=None, **kwargs):
        """Computes the following elements and stores them in unit['analysis']['spatial_ratemaps'],
        overwriting any previous value.

        'spike_rates_smoothed' - numpy.ndarray smoothed ratemap
        'spike_rates_adaptive_smoothed' - numpy.ndarray adaptive smoothed ratemap
        'spike_rates_halves' - {'first': numpy.array ratemap of first half of recording,
                                'second': numpy.array ratemap of second half of recording}
        'spike_rates_minutes' - {'odd': numpy.array ratemap based on odd minutes of recording,
                                 'even': numpy.array ratemap based on even minutes of recording}

        :param dict unit: :py:class:`Recording` attribute `unit` element.
        :param numpy.ndarray xy: shape (n_samples, 2) regularly sampled position data
            columns: (x values - horizontal axis, y values - vertical axis)
        :param float bin_size: spatial bin width for binning xy
        :param n_smoothing_bins: number of spatial bins to use for smoothing (sigma for gaussian method)
        :param str smoothing_method: 'gaussian' or 'boxcar
        :param adaptive_smoothing_alpha: alpha value to use for adaptive smoothing
            This value depends on sampling and has a strong influence on results. For traditional
            place cell recordings (1 x 1 m box for 20 minutes) the alpha value 200 works well.
        :param int position_sampling_rate: Hz
        :param numpy.ndarray xy_mask: boolean array specifying which samples of xy are used
            for computing the spatial maps, with True values for position samples to be included.
        :param dict direction_filter_kwargs:
            {
                'direction': numpy.ndarray (n_samples,) with directional values (-pi to pi) for each sample in `xy`
                'direction_bin_centers': list of directional filtering window centers
                'direction_bin_width': float specifying width of directional window in same units as `direction`
            }
        :param kwargs: any additional keyword arguments are passed on to :py:class:`SpatialRatemap`
        """

        ratemap = SpatialRatemap(xy, unit['timestamps'], position_sampling_rate,
                                 bin_size, xy_mask=xy_mask, **kwargs)

        spatial_window = ratemap.spatial_window
        if 'spatial_window' in kwargs:
            del kwargs['spatial_window']

        unit['analysis']['spatial_ratemaps'] = {
            'bin_size': np.array(bin_size),
            'n_smoothing_bins': np.array(n_smoothing_bins),
            'smoothing_method': smoothing_method,
            'adaptive_smoothing_alpha': np.array(adaptive_smoothing_alpha),
            'spatial_window': np.array(ratemap.spatial_window)
        }

        unit['analysis']['spatial_ratemaps']['spike_rates_smoothed'] = \
            ratemap.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method)

        ratemap_first_half, ratemap_second_half = SpatialRatemap.instantiate_for_first_last_half(
            xy, unit['timestamps'], position_sampling_rate, bin_size, spatial_window=spatial_window,
            xy_mask=xy_mask, **kwargs
        )

        unit['analysis']['spatial_ratemaps']['spike_rates_halves'] = {
            'first': ratemap_first_half.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method),
            'second': ratemap_second_half.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method)
        }

        ratemap_odd_minutes, ratemap_even_minutes = SpatialRatemap.instantiate_for_odd_even_minutes(
            xy, unit['timestamps'], position_sampling_rate, bin_size, spatial_window=spatial_window,
            xy_mask=xy_mask, **kwargs
        )

        unit['analysis']['spatial_ratemaps']['spike_rates_minutes'] = {
            'odd': ratemap_odd_minutes.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method),
            'even': ratemap_even_minutes.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method)
        }

        if not (direction_filter_kwargs is None):
            unit['analysis']['spatial_ratemaps']['direction_filtered_ratemaps'] = \
                SpatialRatemap.compute_direction_filtered_ratemaps(
                    **direction_filter_kwargs, xy=xy, timestamps=unit['timestamps'], bin_size=bin_size,
                    position_sampling_rate=position_sampling_rate,
                    spatial_window=spatial_window, n_smoothing_bins=n_smoothing_bins,
                    smoothing_method=smoothing_method,
                    xy_mask=xy_mask
                )

    @staticmethod
    def compute_all_ratemaps_for_all_units_in_recording(recording, bin_size, n_smoothing_bins,
                                                        smoothing_method, adaptive_smoothing_alpha,
                                                        verbose=False, **kwargs):
        """Applies :py:func:`SpatialRatemap.compute_all_ratemaps_for_unit` on all units in provided
        :py:class:`..recording_io.Recording` instance.

        See :py:func:`SpatialRatemap.compute_all_ratemaps_for_unit` for more details

        :param recording: :py:class:`..recording_io.Recording` instance
        :param float bin_size: spatial bin width for binning xy
        :param n_smoothing_bins: number of spatial bins to use for smoothing (sigma for gaussian method)
        :param str smoothing_method: 'gaussian' or 'boxcar
        :param adaptive_smoothing_alpha: alpha value to use for adaptive smoothing
            This value depends on sampling and has a strong influence on results. For traditional
            place cell recordings (1 x 1 m box for 20 minutes) the alpha value 200 works well.
        :param bool verbose: if True (default) progress is printed out
        :param kwargs: any additional keyword arguments are passed on to :py:class:`SpatialRatemap`
        """
        print('Computing ratemaps for {}'.format(recording.fpath))
        for unit in (tqdm(recording.units) if verbose else recording.units):
            SpatialRatemap.compute_all_ratemaps_for_unit(
                unit, recording.position['xy'], bin_size, n_smoothing_bins,
                smoothing_method, adaptive_smoothing_alpha,
                recording.position['sampling_rate'],
                **kwargs
            )

    @staticmethod
    def compute_ratemap_for_all_units_across_recordings(
            recordings, bin_size, n_smoothing_bins, smoothing_method, name='recordings_spatial_ratemaps',
            xy_masks=None, return_xy_shifted_masked_position_bin_inds=False,
            limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds=None,
            verbose=False, **kwargs):
        """Computes smoothed ratemap with data concatenated across all recordings in
        a :py:class:`..recording_io.Recordings` instance.

        Output is stored at the recordings[0].analysis[name] as a list for all elements in `recordings.units`,
        where `name` is the input string to this method.

        :param recordings: :py:class:`..recording_io.Recordings` instance
        :param float bin_size: spatial bin width for binning xy
        :param n_smoothing_bins: number of spatial bins to use for smoothing (sigma for gaussian method)
        :param str smoothing_method: 'gaussian' or 'boxcar
        :param str name: key for output of this method in unit['analysis'] (default: 'recordings_spatial_ratemaps')
        :param list xy_masks: list of xy_masks for each recording
        :param bool return_xy_shifted_masked_position_bin_inds: if True (default is False) the method returns
            a list of arrays :py:attr:`SpatialRatemap.xy_shifted_masked_position_bin_inds` for each recording.
        :param numpy.ndarray limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds:
            See :py:class:`SpatialRatemap` description.
        :param bool verbose: if True (default) progress is printed out
        :param kwargs: any additional keyword arguments are passed on to :py:class:`SpatialRatemap`
        """

        xy = []
        for recording in recordings:
            xy.append(recording.position['xy'])
        xy = np.concatenate(xy, axis=0)

        if xy_masks is None:
            xy_mask = None
        else:
            xy_mask = np.concatenate(xy_masks, axis=0)

        recordings[0].analysis[name] = []

        ratemap = None
        for i_unit in (tqdm(range(len(recordings.units))) if verbose else range(len(recordings.units))):

            spiketimes = recordings.unit_timestamps_concatenated_across_recordings(i_unit)

            ratemap = SpatialRatemap(
                xy, spiketimes, recordings[0].position['sampling_rate'], bin_size, xy_mask=xy_mask,
                limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds=(
                    limit_xy_mask_to_previous_xy_shifted_masked_position_bin_inds
                ),
                **kwargs)

            recordings[0].analysis[name].append({
                'spike_rates_smoothed': ratemap.spike_rates_smoothed(n_smoothing_bins, method=smoothing_method),
                'bin_size': np.array(bin_size),
                'n_smoothing_bins': np.array(n_smoothing_bins),
                'smoothing_method': smoothing_method,
                'spatial_window': np.array(ratemap.spatial_window)
            })

        if return_xy_shifted_masked_position_bin_inds:
            return ratemap.xy_shifted_masked_position_bin_inds
