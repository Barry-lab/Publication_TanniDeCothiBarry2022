
import numpy as np
import psutil
from scipy.special import factorial
from tqdm import tqdm
from numba import njit, prange
from scipy.ndimage import gaussian_filter1d


from barrylab_ephys_analysis.recording_io import Recording, Recordings
from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap
from barrylab_ephys_analysis.blea_utils import batch_generator
from barrylab_ephys_analysis.spikes.utils import count_spikes_in_sample_bins


class FlatPriorBayes(object):
    """
    Provides Flat Prior Bayes Posterior Probability array based on neuronal tuning vectors
    and observed neuronal population spike values.

    For each sample in population activity (`spike_vectors`), computes the probability
    of observing that population activity in each state covered by `tuning_vectors`.

    `tuning_vectors` and `spike_vectors` must be in same units of measurement
    e.g. both as spike counts per same time-window or as spike rates (Hz).

    The computation is based on formulation in Zhang et al 1998 equation 35,
    with the exception that inputs are already in same time-base, e.g. spike rates.
    https://doi.org/10.1152/jn.1998.79.2.1017

    After instantiation, use :py:func:`FlatPriorBayes.get_posteriors` or
    :py:func:`FlatPriorBayes.get_posteriors_batched` to retrieve posteriors for an array of spike_vectors.
    The latter can compute posteriors separately for sets of samples in spike_vectors
    to avoid running out of memory.
    """

    def __init__(self, tuning_vectors, dtype=np.float64):
        """Instantiates Flat Prior Bayes Posterior Probability estimator

        :param numpy.ndarray tuning_vectors: shape (n_states, n_units)
        :param numpy.float dtype: specify dtype to use, e.g. numpy.float32 or numpy.float64 for lower memory demand.
            numpy.float128 is much slower as some operations can no longer be done parallel.
        """
        self._dtype = dtype

        self._tuning_vectors = tuning_vectors
        if self._tuning_vectors != self._dtype:
            self._tuning_vectors = self._tuning_vectors.astype(self._dtype)

        # Add smallest numpy.float64 value to tuning vectors to avoid 0 values.
        self._tuning_vectors = self._tuning_vectors + np.finfo(self._dtype).resolution

        self._tuning_exponent = None

    @property
    def tuning_vectors(self):
        return self._tuning_vectors

    @property
    def tuning_exponent(self):
        if self._tuning_exponent is None:
            self._tuning_exponent = np.exp(-self.tuning_vectors)
        return self._tuning_exponent

    @staticmethod
    @njit(parallel=True)
    def posterior_computation_power(tuning_vectors, spike_vectors):
        return np.power(tuning_vectors, spike_vectors)

    @staticmethod
    def posterior_computation_power_float128(tuning_vectors, spike_vectors):
        return np.power(tuning_vectors, spike_vectors)

    @staticmethod
    def factorial(x):
        if x.dtype == np.longdouble:
            return factorial(x.astype(np.float64), exact=False).astype(x.dtype)
        else:
            return factorial(x, exact=False)

    @staticmethod
    @njit(parallel=True)
    def posterior_computation_product_not_float128(posterior_working, out):
        for i in prange(posterior_working.shape[0]):
            for j in range(posterior_working.shape[1]):
                out[i, j] = np.prod(posterior_working[i, j, :])
        return out

    @staticmethod
    def posterior_computation_product_float128(posterior_working, out):
        for i in prange(posterior_working.shape[0]):
            for j in range(posterior_working.shape[1]):
                out[i, j] = np.prod(posterior_working[i, j, :])
        return out

    @staticmethod
    def posterior_computation_product(posterior_working):
        """Computes the posterior. Uses a slower method for numpy.float128

        :param numpy.ndarray posterior_working: shape (n_states, n_samples, n_units)
        :return: posterior shape (n_states, n_samples)
        :rtype: numpy.ndarray
        """
        empty_posterior = np.empty((posterior_working.shape[0], posterior_working.shape[1]),
                                   dtype=posterior_working.dtype)
        if posterior_working.dtype == np.longdouble:
            return FlatPriorBayes.posterior_computation_product_float128(posterior_working, empty_posterior)
        else:
            return FlatPriorBayes.posterior_computation_product_not_float128(posterior_working, empty_posterior)

    @staticmethod
    def single_sample_posterior_product_avoid_zero(sample_posterior):
        """Computes the posterior while attempting to avoid any states with all zero-values for a single sample.

        :param numpy.ndarray sample_posterior: array shape (n_states, n_units) to compute product of over n_units.
        :return: posterior shape (n_states,)
        :rtype: numpy.ndarray
        """

        min_multiplier = (np.power(np.finfo(sample_posterior.dtype).resolution, 1. / float(sample_posterior.shape[1]))
                          / np.min(sample_posterior))
        max_multiplier = (np.power(np.finfo(sample_posterior.dtype).max, 1. / float(sample_posterior.shape[1]))
                          / np.max(sample_posterior))

        if min_multiplier > 1 > max_multiplier:
            raise Exception('Unable to compute a suitable multiplier')
        elif min_multiplier > 1:
            sample_posterior = sample_posterior * max_multiplier
        elif max_multiplier < 1:
            sample_posterior = sample_posterior * min_multiplier
        else:
            sample_posterior = sample_posterior * ((min_multiplier + max_multiplier) / 2.)

        posterior = FlatPriorBayes.posterior_computation_product(sample_posterior[:, np.newaxis, :]).flatten()

        if np.any(posterior > 0):
            return posterior
        else:
            raise Exception('Unable to avoid all zero sample posterior')

    @staticmethod
    def posterior_computation_product_avoid_zero(posterior_working):
        """Computes the posterior while attempting to avoid any states with all zero-values.

        :param numpy.ndarray posterior_working: array shape (n_states, n_samples, n_units) to compute
            product of over n_units.
        :return: posterior shape (n_states, n_samples)
        :rtype: numpy.ndarray
        """

        posterior = FlatPriorBayes.posterior_computation_product(posterior_working)

        zero_samples = np.where(np.any(posterior == np.array(0, dtype=posterior.dtype), axis=0))[0]

        if len(zero_samples) == 0:
            return posterior

        for n_sample in zero_samples:

            posterior[:, n_sample] = \
                FlatPriorBayes.single_sample_posterior_product_avoid_zero(posterior_working[:, n_sample, :])

        return posterior

    def get_posteriors(self, spike_vectors, normalise=True, multiplier=None, avoid_zero_posterior=False):
        """Returns the posterior probability matrix shape, specifying the probability
        of observing each sample in `spike_vectors` at each state.

        Posterior values are divided by `n_units` based on `spike_vectors.shape[1]`, unless `normalise=True`.

        :param numpy.ndarray spike_vectors: shape (n_samples, n_units)
        :param bool normalise: if True (default), output columns (each sample) is normalised to sum to 1.
        :param float multiplier: if provided probability contributions of individual units are multiplied
            by this value before computing their product. This can be useful if posterior values are too close to 0.
            Note! This changes posterior values.
        :param bool avoid_zero_posterior: if True (default is False), an attempt is made to avoid all zero values
            in any posterior state.
            Note! This changes posterior values differently for each state. This works independently of `multiplier`.
        :return: posterior_probability shape (n_states, n_samples)
        :rtype: numpy.ndarray
        """
        n_units = spike_vectors.shape[1]

        # Ensure input is in the right shape
        if spike_vectors.shape[1] != self.tuning_vectors.shape[1]:
            raise ValueError('spike_vectors n_units does not match tuning_vectors n_units')
        # Ensure input is self._dtype
        if spike_vectors.dtype != self._dtype:
            spike_vectors = spike_vectors.astype(self._dtype)

        # Reshape tuning_vectors and spike_vectors to expected shape (n_states, n_samples, n_units)
        tuning_vectors = self.tuning_vectors.reshape((self.tuning_vectors.shape[0], 1, self.tuning_vectors.shape[1]))
        spike_vectors = spike_vectors.reshape((1, spike_vectors.shape[0], self.tuning_vectors.shape[1]))

        # Compute top of the fraction, shape (n_states, n_samples, n_units)
        tuning_vectors = np.broadcast_to(tuning_vectors,
                                         (tuning_vectors.shape[0], spike_vectors.shape[1], spike_vectors.shape[2]))
        spike_vectors_big = np.broadcast_to(spike_vectors,
                                            (tuning_vectors.shape[0], spike_vectors.shape[1], spike_vectors.shape[2]))
        if self._dtype is np.longdouble:
            posterior = self.posterior_computation_power_float128(tuning_vectors, spike_vectors_big)
        else:
            posterior = self.posterior_computation_power(tuning_vectors, spike_vectors_big)
        del spike_vectors_big
        del tuning_vectors

        # Compute the division with factorial of spike_vectors, shape (n_states, n_samples, n_units)
        posterior = np.divide(
            posterior,
            self.factorial(spike_vectors),
            out=posterior
        )

        # Compute multiplication with tuning_exponent, shape (n_states, n_samples, n_units)
        posterior = np.multiply(
            posterior,
            self.tuning_exponent.reshape((posterior.shape[0], 1, posterior.shape[2])),
            out=posterior
        )

        # Increase posterior values to avoid 0 in product
        if not (multiplier is None):
            posterior = posterior * multiplier

        # Compute the product across n_units, shape (n_states, n_samples)
        if avoid_zero_posterior:
            posterior = self.posterior_computation_product_avoid_zero(posterior)
        else:
            posterior = self.posterior_computation_product(posterior)

        if normalise:
            # Normalise output columns to sum to 1, if requested, shape (n_states, n_samples)
            posterior = np.divide(
                posterior,
                np.nansum(posterior, axis=0, dtype=posterior.dtype, keepdims=True),
                out=posterior
            )
        else:
            # Otherwise compute posterior values as average per unit
            posterior = np.divide(posterior, np.array(n_units, dtype=posterior.dtype), out=posterior)

        return posterior

    @staticmethod
    def specify_batch_size(n_states, n_units, dtype, margin=0.25):
        """Returns the maximum decoding batch size possible with currently available virtual memory

        :param int n_states: number of possible states (length of tuning_vector per unit)
        :param int n_units: number of units (length of tuning_vector per state)
        :param numpy.float dtype: specify dtype to use, e.g. numpy.float64 or numpy.float128 for higher precision
        :param float margin: proportion (1 - 1.0) of available memory to reserve for other processes
        :return: batch_size
        :rtype: int
        """
        available_memory = (1. - margin) * psutil.virtual_memory().available
        return int(available_memory / float(n_states * n_units * dtype(1.).nbytes))

    def get_posteriors_batched(self, spike_vectors, normalise=True, batch_size=None,
                               multiplier=None, avoid_zero_posterior=False, verbose=False):
        """Returns the posterior probability matrix shape, specifying the probability
        of observing each sample in `spike_vectors` at each state.

        Same as :py:func:`FlatPriorBayes.get_posteriors`, but performs computation separately for
        batches of spike_vectors to avoid running out of memory.

        :param numpy.ndarray spike_vectors: shape (n_samples, n_units)
        :param bool normalise: if True (default), output columns (each sample) is normalised to sum to 1.
        :param int batch_size: if provided, sets the number of samples for which the posteriors
            are computed simultaneously. Otherwise the batch size is inferred from available virtual memory.
        :param float multiplier: if provided probability contributions of individual units are multiplied
            by this value before computing their product. This can be useful if posterior values are too close to 0.
            Note! This changes posterior values.
        :param bool avoid_zero_posterior: if True (default is False), an attempt is made to avoid all zero values
            in any posterior state.
            Note! This changes posterior values differently for each state. This works independently of `multiplier`.
        :param bool verbose: if True, progress is printed in console (default is False).
        :return: posterior_probability shape (n_states, n_samples)
        :rtype: numpy.ndarray
        """
        posteriors = np.ones((self.tuning_vectors.shape[0], spike_vectors.shape[0]),
                             dtype=self._dtype) * np.nan

        iterable = list(batch_generator(
            spike_vectors.shape[0],
            batch_size if not (batch_size is None) else self.specify_batch_size(self.tuning_vectors.shape[0],
                                                                                spike_vectors.shape[0],
                                                                                self._dtype)
        ))

        for (first_ind, last_ind) in (tqdm(iterable) if verbose else iterable):
            posteriors[:, first_ind:last_ind] = \
                self.get_posteriors(spike_vectors[first_ind:last_ind, :], normalise=normalise,
                                    multiplier=multiplier, avoid_zero_posterior=avoid_zero_posterior)

        return posteriors

    @staticmethod
    def find_true_if_many_choose_randomly_among_true_values(x):
        """Returns the index of a True element in 1-D array.
        If many True elements, one of those is chosen randomly.

        :param numpy.ndarray x: shape (N,)
        :return: index
        :rtype: int
        """
        ind = np.where(x)[0]
        if len(ind) == 1:
            return ind[0]
        elif len(ind) > 1:
            return ind[np.random.choice(len(ind))]
        else:
            raise ValueError('input array must contain a True element')

    def get_posterior_peak_indices(self, spike_vectors, batch_size=None, multiplier=None,
                                   avoid_zero_posterior=True, verbose=False):
        """Returns peaks value state indices for each sample in posterior probability matrix
        as computed with :py:func:`FlatPriorBayes.get_posteriors_batched`.

        If posterior has multiple peaks, one is chosen at random.

        :param numpy.ndarray spike_vectors: shape (n_samples, n_units)
        :param int batch_size: if provided, sets the number of samples for which the posteriors
            are computed simultaneously. Otherwise the batch size is inferred from available virtual memory.
        :param float multiplier: if provided probability contributions of individual units are multiplied
            by this value before computing their product. This can be useful if posterior values are too close to 0.
            Note! This changes posterior values.
        :param bool avoid_zero_posterior: if True (default), an attempt is made to avoid all zero values
            in any posterior state.
            Note! This changes posterior values differently for each state. This works independently of `multiplier`.
        :param bool verbose: if True, progress is printed in console (default is False).
        :return: sample_indices shape (n_samples,)
        :rtype: numpy.ndarray
        """
        posteriors = self.get_posteriors_batched(spike_vectors, normalise=False,
                                                 batch_size=batch_size, multiplier=multiplier,
                                                 avoid_zero_posterior=avoid_zero_posterior, verbose=verbose)
        return np.apply_along_axis(self.find_true_if_many_choose_randomly_among_true_values, 0,
                                   np.max(posteriors, axis=0)[None, :] == posteriors)


class FlatPriorBayesPosteriorMaps(object):
    """
    A wrapper for :py:class:`barrylab_ephys_analysis.decoding.FlatPriorBayes` for computing
    2D posterior probability matrices using ratemaps, rather than 1D posteriors and tuning vectors.
    """

    def __init__(self, ratemaps, batch_size=None):
        """
        :param numpy.ndarray ratemaps: ratemaps to represent expected spiking at positions
            shape (n_ybins, n_xbins, n_units).
        :param int batch_size: if provided, sets the number of samples for which the posteriors
            are computed simultaneously. Otherwise the batch size is inferred from available virtual memory.
        """

        self._ratemaps = ratemaps
        self._batch_size = batch_size

        self._ratemap_shape = self._ratemaps.shape[:2]
        self._n_states = np.multiply(*self._ratemap_shape)
        self._n_units = self._ratemaps.shape[2]

        self._decoder = FlatPriorBayes(
            self._ratemaps.reshape(self._n_states, self._ratemaps.shape[2])
        )

    def reshape_decoder_posterior_to_ratemap_shape(self, posterior):
        """Returns the posterior with the first dimension (n_states) being extended into two dimensions
        according to input ratemap shape, resulting in shape (n_ybins, n_xbins, n_samples)

        :param posterior: shape (n_states, n_samples)
        :return: reshaped_posterior
        :rtype: numpy.ndarray
        """
        return posterior.reshape(self._ratemap_shape[0], self._ratemap_shape[1], posterior.shape[1])

    def get_posteriors(self, spike_vectors, normalise=True, verbose=False):
        """Returns the posterior maps in shape of input ratemaps for all samples of spike vectors.

        :param numpy.ndarray spike_vectors: shape (n_samples, n_units)
        :param bool normalise: if True (default), output columns (each sample) is normalised to sum to 1.
        :param bool verbose: if True, progress is printed in console (default is False).
        :return: posteriors shape (n_ybins, n_xbins, n_samples)
        :rtype: numpy.ndarray
        """
        return self.reshape_decoder_posterior_to_ratemap_shape(
            self._decoder.get_posteriors_batched(
                spike_vectors, normalise=normalise,
                batch_size=self._batch_size, verbose=verbose)
        )


class FlatPriorBayesPosteriorMultipleMaps(object):
    """
    A wrapper for :py:class:`barrylab_ephys_analysis.decoding.FlatPriorBayes` for computing multiple
    2D posterior probability matrices using ratemaps of multiple recordings,
    rather than 1D posteriors and tuning vectors.

    Ratemaps are concatenated into a single tuning_vectors array for the decoder and the posteriors
    are computed concurrently for all environments represented by separate ratemap arrays.
    Therefore, if normalization is requested, this is applied simultaneously across all positions of all environments.
    """

    def __init__(self, ratemaps, batch_size=None):
        """
        :param list ratemaps: list of ratemaps to represent expected spiking at positions.
            Each list element representing spiking ratemaps in an environment, shape (n_ybins, n_xbins, n_units).
            Between elements of the ratemaps list, n_ybins and n_xbins can vary, but n_units must be constant.
        :param int batch_size: if provided, sets the number of samples for which the posteriors
            are computed simultaneously. Otherwise the batch size is inferred from available virtual memory.
        """

        self._ratemaps = ratemaps
        self._batch_size = batch_size

        if not all([rm.shape[2] == self._ratemaps[0].shape[2] for rm in self._ratemaps]):
            raise ValueError('Input ratemaps must have same n_units.')

        self._ratemap_shape = [rm.shape[:2] for rm in self._ratemaps]
        self._n_states = [np.multiply(*shape) for shape in self._ratemap_shape]
        self._n_units = self._ratemaps[0].shape[2]

        # Initialize decoder with multiple ratemaps
        self._decoder = FlatPriorBayes(
            np.concatenate([rm.reshape(ns, self._n_units) for rm, ns in zip(self._ratemaps, self._n_states)],
                           axis=0)
        )

    def reshape_decoder_posterior_to_ratemap_shape(self, posterior):
        """Returns the list of posteriors with the first dimension (n_states) being extended into two dimensions
        according to input ratemap shape, resulting in shape (n_ybins, n_xbins, n_samples)

        :param posterior: shape (n_states, n_samples)
        :return: reshaped_posteriors, list of numpy.ndarray
        :rtype: list
        """
        out = []
        for i in range(len(self._ratemaps)):
            out.append(posterior[sum(self._n_states[:i]):sum(self._n_states[:i]) + self._n_states[i], :])
            out[-1] = out[-1].reshape(self._ratemap_shape[i][0], self._ratemap_shape[i][1], out[-1].shape[1])

        return out

    def get_posteriors(self, spike_vectors, normalise=True, verbose=False):
        """Returns the posterior maps in shape of input ratemaps for all samples of spike vectors.

        :param numpy.ndarray spike_vectors: shape (n_samples, n_units)
        :param bool normalise: if True (default), output columns (each sample) is normalised to sum to 1.
        :param bool verbose: if True, progress is printed in console (default is False).
        :return: list of posteriors with shape (n_ybins, n_xbins, n_samples)
        :rtype: list
        """
        return self.reshape_decoder_posterior_to_ratemap_shape(
            self._decoder.get_posteriors_batched(
                spike_vectors, normalise=normalise,
                batch_size=self._batch_size, verbose=verbose)
        )


class FlatPriorBayesPositionDecoding(object):
    """
    A wrapper for :py:class:`barrylab_ephys_analysis.decoding.FlatPriorBayesPosteriorMaps`
    for computing 2D posterior probability matrices conveniently for segments of a recording in
    a :py:class:`recording_io.Recordings` or :py:class:`recording_io.Recording` instance.

    Can compute posteriors for multiple environments from different recordings simultaneously,
    if `recordings` is a :py:class:`recording_io.Recordings` instance and `single_environment` is False.

    Can compute posteriors using cross-validation if `cv_segment_size` is specified.

    Use :py:func:`FlatPriorBayesPositionDecoding.posteriors_for_recording` or
    :py:func:`FlatPriorBayesPositionDecoding.posteriors_for_recordings` for computing the posteriors
    after instantiating this class.
    """

    def __init__(self, recordings, decoding_window_size, decoding_window_overlap,
                 xy_masks=None, single_environment=True, category=None, ratemaps=None,
                 ratemap_kwargs=None, cv_segment_size=None, normalise=True, unit_indices=None, verbose=False):
        """
        :param recordings: either :py:class:`recording_io.Recordings` or :py:class:`recording_io.Recording`
        :param int decoding_window_size: size of decoding window in number of position samples. Must be odd.
        :param int decoding_window_overlap: overlap of decoding window in number of position samples
        :param xy_masks: numpy.ndarray, if :py:class:`recording_io.Recording` provided, or list of numpy.ndarray
            if :py:class:`recording_io.Recordings` is provided as `recordings` argument. Each array should be
            boolean array, shape (n_samples,), with True values for elements of recording.position['xy'] to use
            for creating ratemaps (if not provided) as well as specifying periods to decode.
            `xy_masks[i]` should correspond to `recordings[i].position['xy']`.
        :param bool single_environment: if True (default), all recordings are assumed to be from the same environment
            and for decoding of location the ratemaps are computed by combining data from all recordings.
            If False, the recordings are assumed to be from different environments and separate ratemaps are computed
            for each of recording. In this case, the posteriors are computed simultaneously for all environments,
            regardless of which recording is the source of the spike_vector. This changes the output of some of the
            methods, as multiple posteriors are returned for each decoded time point.
        :param str category: if provided, only units with matching value at unit['analysis']['category'] are included
        :param ratemaps: numpy.ndarray shape (n_ybins, n_xbins, n_units) if `single_environment` is True, otherwise
            list of numpy.ndarray of same shape for each recording in :py:class:`recording_io.Recordings` instance.
            If provided, these ratemaps are used instead of computing ratemaps each time segment is decoded.
            Note! Ratemap values must be in Hz (spikes/second) as spike_vectors for decoding are computed in Hz,
            If this is not provided, each recording instance must contain recording.info['arena_size'] indicating
            the area for which the ratemaps are to be computed (arena_width_x, arena_height_y)
        :param dict ratemap_kwargs: if `ratemaps` is not provided, this dictionary is required for computing
            the ratemaps and must contain the following: {'bin_size': int,
                                                          'n_smoothing_bins': int, (only if smoothing required)
                                                          'smoothing_method': str, (only if smoothing required)
                                                          'n_samples': int (optional)},
            as required by :py:class:`barrylab_ephys_analysis.spatial.ratemaps.SpatialRatemap`.
        :param int cv_segment_size: if provided, decoding is performed using cross-validation,
            where each segment with length cv_segment_size is decoded using ratemaps computed
            based on data outside of that segment. Number of position samples per segment.
        :param bool normalise: if True (default), output columns (each sample) is normalised to sum to 1.
        :param list unit_indices: list of indices of units to use, based on Recordings or Recording ordering,
            depending on which was provided during instantiation. Default is None, meaning all.
            Note: category argument can still override unit_indices.
        :param bool verbose: if True, progress is displayed in console. Default is False.
        """

        if not isinstance(recordings, (Recording, Recordings)):
            raise ValueError('Unknown input type {}'.format(type(recordings)))

        self._recordings = recordings
        if decoding_window_size % 2 == 0:
            raise ValueError('decoding_window_size must be odd')
        self._decoding_window_size = decoding_window_size
        if decoding_window_overlap >= self._decoding_window_size:
            raise ValueError('decoding_window_overlap can not be greater than _decoding_window_size')
        self._decoding_window_steps = self._decoding_window_size - decoding_window_overlap
        self._xy_masks = xy_masks
        self._single_environment = single_environment
        self._category = category
        if not (ratemaps is None) and not (ratemap_kwargs is None):
            raise Exception('Either ratemaps or ratemap_kwargs must be provided.')
        self._ratemaps = ratemaps
        self._ratemap_kwargs = ratemap_kwargs
        if not (cv_segment_size is None) and not (ratemaps is None):
            raise Exception('It does not make sense to use cross-validation with pre-computed ratemaps.')
        self._cv_segment_size = cv_segment_size
        self._normalise = normalise
        self._unit_indices = unit_indices
        self._verbose = verbose

        # Create full_ratemaps memory for each environment that is used when single_environment is False
        if not self.single_environment:
            self._full_ratemaps = None if isinstance(self._recordings, Recording) else [None] * len(self._recordings)
            self._xy_in_sampled_bins = \
                None if isinstance(self._recordings, Recording) else [None] * len(self._recordings)

    @property
    def single_environment(self):
        """Returns True, if class is in mode of decoding position in a single environment (Recording instance)
        """
        return self._single_environment

    def spatial_window(self, i_recording=None):
        if isinstance(self._recordings, Recording):
            return (0, self._recordings.info['arena_size'][0],
                    0, self._recordings.info['arena_size'][1])
        else:
            return (0, self._recordings[i_recording].info['arena_size'][0],
                    0, self._recordings[i_recording].info['arena_size'][1])

    def ratemaps_excluding_segment_recording_instance(self, xy_mask_edit_start_ind, xy_mask_edit_end_ind):

        # Get or create xy_mask for this recording and set the segment part to False, excluding it from ratemap
        xy_mask = (np.ones(self._recordings.position['xy'].shape[0], dtype=np.bool)
                   if self._xy_masks is None else self._xy_masks.copy())
        xy_mask[xy_mask_edit_start_ind:xy_mask_edit_end_ind] = False

        # Iterate over all units to create a ratemap stack
        ratemaps = []
        spatial_ratemaps = None
        # If unit_indices is provided, limit iteration to those units
        iterable = (self._recordings.units
                    if self._unit_indices is None
                    else [self._recordings.units[i] for i in self._unit_indices])
        for unit in iterable:

            # If category specified, ignore units in other categories
            if not (self._category is None) and not (unit['analysis']['category'] in self._category):
                continue

            # Compute the ratemap for this unit and append to list that will be turned into a stack
            spatial_ratemaps = SpatialRatemap(
                self._recordings.position['xy'], unit['timestamps'],
                self._recordings.position['sampling_rate'], self._ratemap_kwargs['bin_size'],
                spatial_window=self.spatial_window(),
                xy_mask=xy_mask,
                n_samples=(self._ratemap_kwargs['n_samples'] if 'n_samples' in self._ratemap_kwargs else None)
            )
            if 'n_smoothing_bins' in self._ratemap_kwargs and 'smoothing_method' in self._ratemap_kwargs:
                ratemaps.append(spatial_ratemaps.spike_rates_smoothed(n_bins=self._ratemap_kwargs['n_smoothing_bins'],
                                                                      method=self._ratemap_kwargs['smoothing_method']))
            else:
                ratemaps.append(spatial_ratemaps.spike_rates)

        return np.stack(ratemaps, axis=2), spatial_ratemaps.xy_in_sampled_bins

    def ratemaps_excluding_segment_recordings_single_environment(self, i_recording, xy_mask_edit_start_ind,
                                                                 xy_mask_edit_end_ind):

        # Ensure arena_size is the same in all recordings, as the ratemaps are computed for the single environment
        if not all([np.all(self._recordings[0].info['arena_size'] == recording.info['arena_size'])
                    for recording in self._recordings]):
            raise ValueError("position['sampling_rate'] must match for all recordings.")
        # Ensure sampling_rate is the same in all recordings, as position data is concatenated for ratemap computation
        if not all([self._recordings[0].position['sampling_rate'] == recording.position['sampling_rate']
                    for recording in self._recordings]):
            raise ValueError("info['arena_size'] must match for all recordings.")

        # Get the xy data concatenated across all recordings
        xy = self._recordings.position_data_concatenated_across_recordings()['xy']

        # Get or create xy_masks for all recordings
        xy_masks = ([xy_mask.copy() for xy_mask in self._xy_masks] if isinstance(self._xy_masks, list)
                    else [np.ones(recording.position['xy'].shape[0], dtype=np.bool)
                          for recording in self._recordings])
        # Set the segment part of the xy_mask to False, excluding it from ratemap, for recording where the segment is
        xy_masks[i_recording][xy_mask_edit_start_ind:xy_mask_edit_end_ind] = False
        # Concatenate all xy_masks
        xy_masks = np.concatenate(xy_masks)

        # Iterate over all units to create a ratemap stack
        ratemaps = []
        spatial_ratemaps = None
        for i_unit, unit in enumerate(self._recordings.units):

            # If unit_indices is provided, only continue if the i_unit is in unit_indices list.
            if not (self._unit_indices is None) and not (i_unit in self._unit_indices):
                continue

            # If category specified, ignore units in other categories
            if (not (self._category is None)
                    and not (self._recordings.first_available_recording_unit(i_unit)['analysis']['category']
                         in self._category)):
                continue

            # Get unit spike timestamps concatenated across all recordings
            timestamps = self._recordings.unit_timestamps_concatenated_across_recordings(
                i_unit, position_sample_gap=True
            )

            # Compute the ratemap for this unit and append to list that will be turned into a stack
            spatial_ratemaps = SpatialRatemap(
                xy, timestamps, self._recordings[0].position['sampling_rate'],
                self._ratemap_kwargs['bin_size'],
                spatial_window=self.spatial_window(0),
                xy_mask=xy_masks,
                n_samples=(self._ratemap_kwargs['n_samples'] if 'n_samples' in self._ratemap_kwargs else None)
            )
            if 'n_smoothing_bins' in self._ratemap_kwargs and 'smoothing_method' in self._ratemap_kwargs:
                ratemaps.append(spatial_ratemaps.spike_rates_smoothed(n_bins=self._ratemap_kwargs['n_smoothing_bins'],
                                                                      method=self._ratemap_kwargs['smoothing_method']))
            else:
                ratemaps.append(spatial_ratemaps.spike_rates)

        return np.stack(ratemaps, axis=2), spatial_ratemaps.xy_in_sampled_bins

    def ratemaps_excluding_segment_recordings_multiple_environments(self, i_recording, xy_mask_edit_start_ind,
                                                                    xy_mask_edit_end_ind):

        # Iterate over all recordings to compute ratemap stacks separately for each recording
        all_ratemaps = []
        xy_in_sampled_bins = []
        for i, recording in enumerate(self._recordings):

            # If the segment is not in this recording and ratemap stack has already been computed, use that stack
            if i != i_recording and not (self._full_ratemaps[i] is None):
                all_ratemaps.append(self._full_ratemaps[i])
                xy_in_sampled_bins.append(self._xy_in_sampled_bins[i])
                continue

            # Get or create xy_mask for this recording
            xy_mask = (np.ones(recording.position['xy'].shape[0], dtype=np.bool)
                       if self._xy_masks is None else self._xy_masks[i].copy())
            # If the segment is in this recording, set the segment part to False, excluding it from ratemap
            if i == i_recording:
                xy_mask[xy_mask_edit_start_ind:xy_mask_edit_end_ind] = False

            # Iterate over all units, based on Recordings instance level unit identities, to create a ratemap stack
            ratemaps = []
            spatial_ratemaps = None
            for i_unit, unit in enumerate(self._recordings.units):

                # If unit_indices is provided, only continue if the i_unit is in unit_indices list.
                if not (self._unit_indices is None) and not (i_unit in self._unit_indices):
                    continue

                # If category specified, ignore units in other categories
                if (not (self._category is None)
                        and not (self._recordings.first_available_recording_unit(i_unit)['analysis']['category']
                             in self._category)):
                    continue

                # Compute the ratemap for this unit and append to list that will be turned into a stack
                # If unit is None, then ratemap is created with no spikes, i.e. all zero ratemap
                spatial_ratemaps = SpatialRatemap(
                    self._recordings[i].position['xy'],
                    np.array([], dtype=np.float64) if unit[i] is None else unit[i]['timestamps'],
                    self._recordings[i].position['sampling_rate'], self._ratemap_kwargs['bin_size'],
                    spatial_window=self.spatial_window(i),
                    xy_mask=xy_mask,
                    n_samples=(self._ratemap_kwargs['n_samples'] if 'n_samples' in self._ratemap_kwargs else None)
                )
                if 'n_smoothing_bins' in self._ratemap_kwargs and 'smoothing_method' in self._ratemap_kwargs:
                    ratemaps.append(
                        spatial_ratemaps.spike_rates_smoothed(n_bins=self._ratemap_kwargs['n_smoothing_bins'],
                                                              method=self._ratemap_kwargs['smoothing_method'])
                    )
                else:
                    ratemaps.append(spatial_ratemaps.spike_rates)

            # Add ratemap stack to list of ratemap stacks
            all_ratemaps.append(np.stack(ratemaps, axis=2))
            xy_in_sampled_bins.append(spatial_ratemaps.xy_in_sampled_bins)

            # If the segment was not in this recording, place this ratemap stack to memory of stacks for recordings
            if i != i_recording:
                self._full_ratemaps[i] = all_ratemaps[i]
                self._xy_in_sampled_bins[i] = xy_in_sampled_bins[i]

        return all_ratemaps, xy_in_sampled_bins

    def number_of_position_samples(self, i_recording):
        if isinstance(self._recordings, Recording):
            return self._recordings.position['xy'].shape[0]
        else:  # Then must be Recordings instance as only these two are allowed in __init__ method
            return self._recordings[i_recording].position['xy'].shape[0]

    def ratemaps_excluding_segment(self, i_recording, segment_start, segment_end):
        """Returns ratemaps for this decoding while excluding this particular segment from computation.

        :param int i_recording: recording index of the segment to be ignored in computing ratemaps
        :param int segment_start: beginning of the segment to be ignored in computing ratemaps
        :param int segment_end: end of the segment to be ignored in computing ratemaps
        :return: (ratemaps_array if `single_environment` is True, otherwise list_of_ratemap_arrays,
                  xy_in_sampled_bins bool vector if `single_environment` is True, otherwise xy_in_sampled_bins vectors)
        """
        xy_mask_edit_start_ind = segment_start - self._decoding_window_size
        xy_mask_edit_start_ind = xy_mask_edit_start_ind if xy_mask_edit_start_ind > 0 else 0
        xy_mask_edit_end_ind = segment_end + self._decoding_window_size
        xy_mask_edit_end_ind = (xy_mask_edit_end_ind
                                if xy_mask_edit_end_ind < self.number_of_position_samples(i_recording)
                                else self.number_of_position_samples(i_recording))

        if isinstance(self._recordings, Recording):

            return self.ratemaps_excluding_segment_recording_instance(xy_mask_edit_start_ind, xy_mask_edit_end_ind)

        else:  # Then must be Recordings instance as only these two are allowed in __init__ method

            if self.single_environment:

                return self.ratemaps_excluding_segment_recordings_single_environment(i_recording,
                                                                                     xy_mask_edit_start_ind,
                                                                                     xy_mask_edit_end_ind)

            else:  # If recordings are not in a single environment, ratemaps are computed separately for each recording

                return self.ratemaps_excluding_segment_recordings_multiple_environments(i_recording,
                                                                                        xy_mask_edit_start_ind,
                                                                                        xy_mask_edit_end_ind)

    def ratemaps_excluding_segment_or_available_ratemaps(self, i_recording, segment_start, segment_end):
        """Returns ratemaps provided during instantiation of this class or computes them whilst
        excluding the segment of data specified.

        :param int i_recording: recording index of the segment to be ignored in computing ratemaps
        :param int segment_start: beginning of the segment to be ignored in computing ratemaps
        :param int segment_end: end of the segment to be ignored in computing ratemaps
        :return: (ratemaps_array if `single_environment` is True, otherwise list_of_ratemap_arrays,
                  xy_in_sampled_bins bool vector if `single_environment` is True, otherwise xy_in_sampled_bins vectors)
                  Note! xy_in_sampled_bins is None, if ratemaps are provided during instantiation.
        """
        if self._ratemaps is None:
            return self.ratemaps_excluding_segment(i_recording, segment_start, segment_end)
        else:
            return self._ratemaps, None

    def position_samples_in_recording(self, i_recording):
        """Returns number of position samples in specified recording..

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param int i_recording: recording index
        :return: n_samples
        :rtype: int
        """
        return (self._recordings.position[i_recording]['xy'].shape[0]
                if isinstance(self._recordings, Recordings)
                else self._recordings.position['xy'].shape[0])

    def position_indices_of_spike_rate_estimation_windows(self, i_recording, xy_mask=None):
        """Returns position indices for spike rate estimation windows in a specific recording.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param int i_recording: recording index
        :param numpy.ndarray xy_mask: boolean vector of same length as number of position samples
            (or list of boolean arrays `if not self.single_environment`).
            If provided, only position samples matching True values in this array are used for decoding.
            Note! This mask is applied in addition to the `xy_mask` provided during instantiation of this class.
        :return: position_indices for centers of decoding windows.
        :rtype: list
        """

        xy_mask = xy_mask[i_recording] if isinstance(xy_mask, list) else xy_mask

        # Get number of samples in position data in this recording
        n_position_samples = self.position_samples_in_recording(i_recording)

        # Compute required clearance from end of position samples
        window_clearance = int(np.ceil(self._decoding_window_size / 2.))

        window_idx = np.zeros(n_position_samples, dtype=np.bool)
        potential_window_inds = list(range(window_clearance, n_position_samples - window_clearance,
                                           self._decoding_window_steps))
        window_idx[potential_window_inds] = True

        # Filter out positions based on xy_mask if provided
        if not (xy_mask is None):
            window_idx = window_idx & xy_mask

        # Filter out positions outside arena size
        xy = (self._recordings.position['xy'] if isinstance(self._recordings, Recording)
              else self._recordings[i_recording].position['xy'])
        window_idx = window_idx & ((xy[:, 0] > 0)
                                   & (xy[:, 0] < self.spatial_window(i_recording)[1])
                                   & (xy[:, 1] > 0)
                                   & (xy[:, 1] < self.spatial_window(i_recording)[3]))

        # If xy_mask was also provided during instantiation, also filter by that xy_mask
        if not (self._xy_masks is None):

            window_idx = window_idx & (self._xy_masks[i_recording]
                                       if isinstance(self._xy_masks, list)
                                       else self._xy_masks)

        return np.where(window_idx)[0]

    def get_position_timestamps(self, i_recording):
        """Returns position timestamps for specific recording.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param int i_recording: recording index
        :return: timestamps
        :rtype: numpy.ndarray
        """
        if isinstance(self._recordings, Recording):
            return self._recordings.position['timestamps']
        else:  # Then must be Recordings instance as only these two are allowed in __init__ method
            return self._recordings[i_recording].position['timestamps']

    def get_position_bin_edges(self, i_recording, first_position_index, last_position_index):
        """Returns edges of bins centered on positions from first to last position index.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param int i_recording: recording index
        :param int first_position_index:
        :param int last_position_index:
        :return: bin_edges
        :rtype: numpy.ndarray
        """
        timestamps = self.get_position_timestamps(i_recording)[first_position_index:last_position_index]
        bin_half_width = (timestamps[1] - timestamps[0]) / 2.
        return np.concatenate([timestamps - bin_half_width, np.array([timestamps[-1] + bin_half_width])])

    def get_unit_timestamps(self, i_recording):
        """Returns timestamps of all units. If Recordings instance was provided,
        returns timestamps concatenated across recordings. If category was provided,
        units are filtered by category.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param int i_recording: recording index
        :return: list_of_timestamps
        :rtype: list
        """
        timestamps = []
        for i in range(len(self._recordings.units)):

            # If unit_indices is provided, only continue if the i unit is in unit_indices list.
            if not (self._unit_indices is None) and not (i in self._unit_indices):
                continue

            if isinstance(self._recordings, Recording):

                if (not (self._category is None)
                        and not (self._recordings.units[i]['analysis']['category'] in self._category)):
                    continue
                unit = self._recordings.units[i]

            else:  # Then must be Recordings instance as only these two are allowed in __init__ method

                if (not (self._category is None)
                        and not (self._recordings.first_available_recording_unit(i)['analysis']['category']
                             in self._category)):
                    continue
                unit = self._recordings.units[i][i_recording]

            # If unit has no spikes in this recording, include an empty array instead
            if unit is None:
                timestamps.append(np.array([], dtype=np.float64))
            else:
                timestamps.append(unit['timestamps'])

        return timestamps

    @property
    def position_sampling_rate(self):
        """Position sampling rate.

        :return: sampling_rate
        :rtype: int
        """
        if isinstance(self._recordings, Recording):
            return self._recordings.position['sampling_rate']
        else:  # Then must be Recordings instance as only these two are allowed in __init__ method
            return self._recordings[0].position['sampling_rate']

    def spike_vectors(self, i_recording, segment_start, segment_end, xy_mask=None):
        """Returns the an array of spike rates for computing posteriors and
        the center position data indices of these spike rate windows.

        :param int i_recording: recording index
        :param int segment_start: beginning of segment to compute the posterior for, in position indices
        :param int segment_end: end of segment to compute the posterior for, in position indices
        :param numpy.ndarray xy_mask: boolean vector of same length as number of position samples
            (or list of boolean arrays `if not self.single_environment`).
            If provided, only position samples matching True values in this array are used for decoding.
        :return: spike_vectors (n_samples, n_units), position_indices (n_samples,)
        :rtype: numpy.ndarray, numpy.ndarray
        """
        # Get position indices of window centers for spike rate estimation
        position_indices = self.position_indices_of_spike_rate_estimation_windows(i_recording, xy_mask=xy_mask)
        position_indices = position_indices[(position_indices >= segment_start) & (position_indices < segment_end)]

        spike_vectors = []
        for spike_histogram in map(lambda args: count_spikes_in_sample_bins(*args),
                                   [(timestamps, self.position_sampling_rate,
                                     segment_start, segment_end - 1, self._decoding_window_size)
                                    for timestamps in self.get_unit_timestamps(i_recording)]):

            # Only take values for requested position indices, aligning position index values
            # to segment_starts spike_histogram will start from segment_start index
            spike_vectors.append(spike_histogram[position_indices - segment_start, None])

        # Pre-compute the length of the window in seconds for conversion of spike counts to rate in Hz
        window_duration = (1. / self.position_sampling_rate) * self._decoding_window_size

        # Return results concatenated across all units and converted to firing rate in Hz
        return np.concatenate(spike_vectors, axis=1) / window_duration, position_indices

    def segment_posteriors(self, i_recording, segment_start, segment_end, ratemaps,
                           xy_mask=None, batch_size=None):
        """Returns the posteriors for each decoded sample in the specified segment and the corresponding
        position sample indices of decoding window centers.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        In cases if no position indices can be decoded from in range of segment_start and segment_end,
        the method returns a tuple (None, None).

        :param int i_recording: recording index
        :param int segment_start: beginning of segment to compute the posterior for, in position indices
        :param int segment_end: end of segment to compute the posterior for, in position indices
        :param ratemaps: ratemaps to use as tuning vectors of units, numpy.ndarray shape (n_ybins, n_xbins, n_units)
            or if `single_environment` is True, list of such arrays.
        :param numpy.ndarray xy_mask: boolean vector of same length as number of position samples
            (or list of boolean arrays `if not self.single_environment`).
            If provided, only position samples matching True values in this array are used for decoding.
        :param int batch_size: number of position samples to compute simultaneously. Default is None, automatic.
        :return: (posteriors numpy.ndarray shape (n_ybins, n_xbins, n_samples) or
            if `single_environment` is False, list of such arrays),
            (position_indices (n_samples,) of decoding window centers)
        """
        if self.single_environment:
            decoder = FlatPriorBayesPosteriorMaps(ratemaps, batch_size=batch_size)
        else:
            decoder = FlatPriorBayesPosteriorMultipleMaps(ratemaps, batch_size=batch_size)

        spike_vectors, position_indices = self.spike_vectors(i_recording, segment_start, segment_end, xy_mask=xy_mask)

        if position_indices.size == 0:
            return None, None
        else:
            return (decoder.get_posteriors(spike_vectors, normalise=self._normalise,
                                           verbose=self._verbose if self._cv_segment_size is None else False),
                    position_indices)

    def recording_cross_validation_segments(self, i_recording):
        """Returns an iterable of tuples specifying the (segment_start, segment_end) for each cross-validation fold
        based on `cv_segment_size` parameter for specified recording.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        :param i_recording: recording index
        :return: cross_validation_indices, an iterable of tuples (segment_start, segment_end)
        :rtype: iterable
        """
        return batch_generator(self.position_samples_in_recording(i_recording),
                               self._cv_segment_size)

    def posteriors_for_recording(self, i_recording, batch_size=None):
        """Returns the posteriors for each decoded sample in the specified recording and the corresponding
        position sample indices of decoding window centers.

        If `recordings` argument was :py:class:`recording_io.Recording`, then `i_recording` is ignored.

        If `cv_segment_size` was specified, posteriors are computed using ratemaps computed on data
        excluding the data in segment of size `cv_segment_size`. The output posteriors are then concatenated.

        :param i_recording: recording index
        :param int batch_size: number of decoding windows to compute simultaneously. Default is None, automatic.
        :return: posteriors numpy.ndarray shape (n_ybins, n_xbins, n_samples) or
            if `signle_environment` is False, list of such arrays,
            position_indices (n_samples,) of decoding window centers
        """

        if self._cv_segment_size is None:  # in this case, ratemaps are computed using all data

            ratemaps, xy_in_sampled_bins = self.ratemaps_excluding_segment_or_available_ratemaps(i_recording, 0, 0)

            return self.segment_posteriors(
                i_recording, 0, self.position_samples_in_recording(i_recording),
                ratemaps, xy_in_sampled_bins, batch_size=batch_size
            )

        else:

            # If cv_segment_size is specified, iterate over cross-validation segments
            posteriors = []
            position_indices = []
            iterable = list(self.recording_cross_validation_segments(i_recording))
            for (segment_start, segment_end) in (tqdm(iterable) if self._verbose else iterable):

                ratemaps, xy_in_sampled_bins = \
                    self.ratemaps_excluding_segment_or_available_ratemaps(i_recording, segment_start, segment_end)

                ret1, ret2 = self.segment_posteriors(
                    i_recording, segment_start, segment_end, ratemaps, xy_in_sampled_bins, batch_size=batch_size
                )
                # In some cases the specified segment may not have any position indices that can be included
                # in decoding. In that case, the above method returns (None, None) and this loop iteration is skipped.
                if ret1 is None and ret2 is None:
                    continue

                posteriors.append(ret1)
                position_indices.append(ret2)

            if self.single_environment:  # concatenation of cross-validation segments in this case is straightforward

                return np.concatenate(posteriors, axis=2), np.concatenate(position_indices)

            else:  # concatenation of cross-validation segments must be done separately for each set of posteriors

                return ([np.concatenate([p[i] for p in posteriors], axis=2)
                         for i in range(len(posteriors[0]))],
                        np.concatenate(position_indices))

    def posteriors_for_recordings(self, batch_size=None):
        """Returns the posteriors for each decoded sample in the input `recordings` instance and the corresponding
        position sample indices of decoding window centers.

        :param int batch_size: number of decoding windows to compute simultaneously. Default is None, automatic.
        :return: (posteriors numpy.ndarray shape (n_ybins, n_xbins, n_samples) or
            if `single_environment` is False, list of such arrays and if `recordings` parameter was
            :py:class:`recording_io.Recordings` instance, then output is a list of the above),
            (position_indices (n_samples,) of decoding window centers)
        """

        if isinstance(self._recordings, Recording):

            return self.posteriors_for_recording(0, batch_size=batch_size)

        else:

            return [self.posteriors_for_recording(i_recording, batch_size=batch_size)
                    for i_recording in range(len(self._recordings))]
