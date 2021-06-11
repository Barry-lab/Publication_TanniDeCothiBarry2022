import warnings
import numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit, OptimizeWarning
from copy import deepcopy

from tqdm import tqdm

from barrylab_ephys_analysis.recording_io import Recording, Recordings
from barrylab_ephys_analysis.spikes.correlograms import create_correlation_bin_edges, cross_correlation
from barrylab_ephys_analysis.spatial.similarity import spatial_correlation


def mahalnobis(u, v):
    """
    By Robin Hayman

    gets the mahalanobis distance between two vectors feature arrays u and v
    a blatant copy of the Mathworks fcn as it doesn't require the covariance
    matrix to be calculated which is a pain if there are NaNs in the matrix
    """
    u_sz = u.shape
    v_sz = v.shape
    if u_sz[1] != v_sz[1]:
        warnings.warn('Input size mismatch: matrices must have same number of columns')
    if v_sz[0] < v_sz[1]:
        warnings.warn('Too few rows: v must have more rows than columns')
    if np.any(np.imag(u)) or np.any(np.imag(v)):
        warnings.warn('No complex inputs are allowed')
    m = np.nanmean(v,axis=0)
    M = np.tile(m, reps=(u_sz[0],1))
    C = v - np.tile(m, reps=(v_sz[0],1))
    Q, R = np.linalg.qr(C)
    ri = np.linalg.solve(R.T, (u-M).T)
    d = np.sum(ri * ri,0).T * (v_sz[0]-1)
    return d


def get_l_ratio_and_isolation_distance(features, cluster_idx):
    """Returns the L-ratio and Isolation Distance measures calculated
    from mahalnobis distance on features provided

    :param numpy.ndarray features: shape (n_spikes, n_features)
    :param cluster_idx: shape (n_spikes,) array specifying the spikes
        for which the L-ratio and Isolation Distance measures are required.
    :return: l_ratio, isolation_dist
    :rtype: float, float
    """
    n_spikes_in_cluster = np.count_nonzero(cluster_idx)
    try:
        d = mahalnobis(features,features[cluster_idx, :])
        # get the indices of the spikes not in the cluster
        M_noise = d[~cluster_idx]
        L = np.sum(1 - chi2.cdf(M_noise, features.shape[1]))
        l_ratio = L / n_spikes_in_cluster
        # calculate isolation distance
        if n_spikes_in_cluster < cluster_idx.size / 2:
            M_noise.sort()
            isolation_dist = M_noise[n_spikes_in_cluster]
        else:
            isolation_dist = np.nan
    except Exception as e:
        print(e)
        print('Continuing')
        isolation_dist = l_ratio = np.nan

    return l_ratio, isolation_dist


class Quality(object):
    """
    Creates a continuous feature array and indexing methods for all units provided and has
    some convenience functions to compute cluster sorting measures for Recording and
    Recordings class units.
    """

    def __init__(self, units, feature_names=None):
        """

        Assumes all units have the same features stored in a dictionary
        at unit['analysis']['waveform_properties']

        :param iterable units: list of units as in :py:attr:`recording_io.Recording.units`
            Can be list of units combined across multiple recordings as with
            :py:func:`recordings.get_units_for_one_tetrode_as_one_level_list`
        :param iterable feature_names: keys of features to use from
            `unit['analysis']['waveform_properties']`. These correspond to property
            names of :py:class:`spikes.waveforms.WaveformProperties`
        """

        # Count number of spikes per unit
        self.n_spikes = []
        for unit in units:
            self.n_spikes.append(unit['timestamps'].size if not (unit is None) else 0)

        # Get list of all available features if not specified
        if feature_names is None:
            for unit in units:
                # Find the first unit that is not None to get waveform feature list
                if not (unit is None):
                    feature_names = list(unit['analysis']['waveform_properties'])
                    break

        # Create concatenated feature array across all units and all features
        self.features = []
        for unit in units:
            if not (unit is None):
                self.features.append(
                    np.concatenate(
                        [unit['analysis']['waveform_properties'][name].astype(np.float64)
                         for name in feature_names],
                        axis=1
                    )
                )
        self.features = np.concatenate(self.features, axis=0)

    def get_l_ratio_and_isolation_distance(self, unit_index):
        """Returns the L-ratio and Isolation Distance for the unit or unit group specified.

        :param unit_index: int or list of ints specifying index of unit in unit list provided
            during instantiation. If list of ints, these units are considered to be a single unit.
        :type unit_index: int or list of ints
        :return:
        """
        if isinstance(unit_index, int):
            cluster_idx = self.get_cluster_idx_for_unit(unit_index)
        elif (isinstance(unit_index, list)
              and (all([isinstance(i, int) for i in unit_index])
                   or all([np.issubdtype(i, np.integer) for i in unit_index]))):
            cluster_idx = self.get_cluster_idx_for_units(unit_index)
        else:
            raise ValueError('Unexpected input format.')

        return get_l_ratio_and_isolation_distance(self.features, cluster_idx)

    def get_cluster_idx_for_unit(self, unit_index):
        """Returns a boolean array of shape (`Quality.features.shape[0]`,) with True
        values for rows in `Quality.features` that correspond to the unit specified.

        :param int unit_index: unit index in list of units provided during instantiation
        :return: cluster_idx boolean array
        :rtype: numpy.ndarray
        """
        start_index = sum(self.n_spikes[:unit_index])
        end_index = start_index + self.n_spikes[unit_index]
        cluster_idx = np.zeros(self.features.shape[0], dtype=np.bool)
        cluster_idx[start_index:end_index] = True

        return cluster_idx

    def get_cluster_idx_for_units(self, unit_index_list):
        """Returns a boolean array of shape (`Quality.features.shape[0]`,) with True
        values for rows in `Quality.features` that correspond to the units specified.

        Essentially same as :py:func:`Quality.get_cluster_idx_for_unit` but can be used
        in case of combined recordings where multiple units belong to the same cluster
        in multiple recordings.

        :param list unit_index_list: list of unit indices in list of units provided during instantiation
        :return: cluster_idx boolean array
        :rtype: numpy.ndarray
        """
        cluster_idx = self.get_cluster_idx_for_unit(unit_index_list[0])
        for unit_index in unit_index_list[1:]:
            cluster_idx = np.logical_or(cluster_idx,
                                        self.get_cluster_idx_for_unit(unit_index))

        return cluster_idx

    @staticmethod
    def compute_all_quality_measures_for_units_in_list_in_analysis(units, unit_group_ids=None,
                                                                   feature_names=None):
        """Uses features in unit['analysis']['waveform_properties'] as specified with feature_names
        to compute all quality measures and stores them in unit['analysis']['sorting_quality'].

        Units should either be a list of unit dictionaries as :py:attr:`recording_io.Recording.units`
        or :py:func:`recording_io.Recordings.get_units_for_one_tetrode_as_one_level_list`.
        In the latter case the `units` list should be accompanied by `unit_group_ids` of same length
        that specifies the elements of `units` that should be considered as a single unit with the same
        index.

        Usually this method would be used on a list of units detected on the same tetrode or
        other proximal channel group.

        :Example:

        >>> from barrylab_ephys_analysis.recording_io import Recordings
        >>> from barrylab_ephys_analysis.spikes.sorting import Quality
        >>> recordings = Recordings(['/data1/filename', '/data2/filename'])
        >>> tetrode_units, tetrode_unit_group_ids = /
        >>>     recordings.get_units_for_one_tetrode_as_one_level_list(3)
        >>> Quality.compute_all_quality_measures_for_units_in_list_in_analysis(
        >>>     tetrode_units, unit_group_ids=tetrode_unit_group_ids)

        :param list units: list of unit dictionaries
        :param list unit_group_ids: list of int, specifying units in same group with same int
        :param list feature_names: see :py:class:`spikes.sorting.Quality` parameters
        """

        if unit_group_ids is None:
            unit_group_ids = list(range(len(units)))

        # Only use units that are not None
        unit_group_ids = [i for i, unit in zip(unit_group_ids, units) if not (unit is None)]
        units = [unit for unit in units if not (unit is None)]

        quality = Quality(units, feature_names=feature_names)

        for unit_group_ind in np.unique(unit_group_ids):
            unit_index_list = list(np.where(np.array(unit_group_ids) == unit_group_ind)[0])
            l_ratio, isolation_distance = quality.get_l_ratio_and_isolation_distance(unit_index_list)
            for unit_index in unit_index_list:
                units[unit_index]['analysis']['sorting_quality'] = \
                    {'l_ratio': np.array(l_ratio),
                     'isolation_distance': np.array(isolation_distance)}


def compute_all_quality_measures_for_all_units_in_recordings(recordings, feature_names=None, verbose=False):
    """Computes all sorting quality measures of all tetrodes and
    stores them in `analysis` field of each `unit` dictionary.

    Measures are computed concurrently for the same unit across all individual Recording instances
    in `recordings`. The same unit in each Recording instance will have the same quality measure values.

    See more details in docs of :py:func:`Quality.compute_all_quality_measures_for_units_in_list_in_analysis`.

    :param recordings: :py:class:`recording_io.Recordings` instance
    :param list feature_names: see :py:class:`spikes.sorting.Quality` parameters
    :param bool verbose: if True, prints progress information. Default is False.
    """
    if verbose:
        print('Computing waveform_properties for {} {}'.format(recordings.info[0]['animal'],
                                                               recordings.info[0]['rec_datetime']))
    iterable = recordings.get_list_of_tetrode_nrs_across_recordings()
    for tetrode_nr in (tqdm(iterable) if verbose else iterable):

        tetrode_units, tetrode_unit_group_ids = \
            recordings.get_units_for_one_tetrode_as_one_level_list(tetrode_nr)

        Quality.compute_all_quality_measures_for_units_in_list_in_analysis(
            tetrode_units, unit_group_ids=tetrode_unit_group_ids, feature_names=feature_names
        )


class UnitDuplicateDetector(object):
    """
    Identifies duplicate units within a :py:class:`recording_io.Recordings` or :py:class:`recording_io.Recording`
    instance based on coincidental firing and ratemap similarity.

    Example usage:
    >>> from barrylab_ephys_analysis.spikes.sorting import UnitDuplicateDetector
    >>> from barrylab_ephys_analysis.recording_io import Recordings
    >>> recordings = Recordings(['/path/to/first/recording_file.nwb', '/path/to/second/recording_file.nwb'])
    >>> udd = UnitDuplicateDetector(recordings, verbose=True)
    >>> duplicate_unit_pairs = udd.list_duplicates(0.025, 0.002, 0.0005, 200, 0.5, category='place_cell')
    >>> print(duplicate_unit_pairs)
    """

    def __init__(self, recordings, verbose=False):
        """
        :param recordings: either :py:class:`recording_io.Recordings` or :py:class:`recording_io.Recording`
        :param bool verbose: set verbosity of console messages
        """

        if not isinstance(recordings, (Recording, Recordings)):
            raise ValueError('Unknown input type {}'.format(type(recordings)))

        self._recordings = recordings
        self._verbose = verbose

        self._bin_edges = {}

    @property
    def unit_pairs(self):
        """All unit index pairs in this recording.

        :return: list of tuples
        :rtype: list
        """
        unit_pairs = []
        for i in range(1, len(self._recordings.units)):
            for j in range(i + 1, len(self._recordings.units)):
                unit_pairs.append((i, j))
        return unit_pairs

    def get_unit_timestamps(self, i):
        """Returns timestamps of the specified unit. If Recordings instance was provided,
        returns timestamps concatenated across recordings.

        :param int i: unit index
        :return: timestamps
        :rtype: numpy.ndarray
        """
        if isinstance(self._recordings, Recording):
            return self._recordings.units[i]['timestamps']
        elif isinstance(self._recordings, Recordings):
            return self._recordings.unit_timestamps_concatenated_across_recordings(i)

    def get_unit_category(self, i):
        """Returns the category of specified unit. If Recordings instance was provided,
        returns the category based on the unit category in first recording where it is detected.

        :param int i: unit index
        :return: category
        :rtype: str
        """
        if isinstance(self._recordings, Recording):
            return self._recordings.units[i]['analysis']['category']
        elif isinstance(self._recordings, Recordings):
            return self._recordings.first_available_recording_unit(i)['analysis']['category']

    def bin_edges(self, max_lag, bin_size):
        """Returns query bin edges for specified max_lag and bin_size

        :param float max_lag: maximum time lag to query
        :param float bin_size: bin size of time lags
        :return: bin_edges
        :rtype: numpy.ndarray
        """
        if (max_lag, bin_size) not in self._bin_edges:
            self._bin_edges[(max_lag, bin_size)] = create_correlation_bin_edges(max_lag, bin_size)
        return self._bin_edges[(max_lag, bin_size)]

    def bin_centers(self, max_lag, bin_size):
        """Returns bin centers of bin edges with specified max_lag and bin_size

        :param float max_lag: maximum time lag to query
        :param float bin_size: bin size of time lags
        :return: bin_centers
        :rtype: numpy.ndarray
        """
        return (self.bin_edges(max_lag, bin_size)[:-1]
                + (self.bin_edges(max_lag, bin_size)[1] - self.bin_edges(max_lag, bin_size)[0]) / 2.)

    def compute_coincident_spike_counts(self, i, j, max_lag, bin_size):
        """Returns the number of coincidental firing instances at specified time lags

        :param int i: first unit index
        :param int j: second unit index
        :param float max_lag: maximum time lag to query
        :param float bin_size: bin size of time lags
        :return: coincident_spike_counts
        :rtype: numpy.ndarray
        """
        return cross_correlation(
            self.get_unit_timestamps(i), self.get_unit_timestamps(j),
            self.bin_edges(max_lag, bin_size), normalize=False, counts=True
        )

    @staticmethod
    def histogram_gaussian(x, baseline, height, center, width):
        """Generates values according from a modified standard gaussian,
        with height multiplier and baseline addition

        :param x: query point
        :param float baseline: baseline addition factor
        :param float height: height multiplier factor
        :param float center: peak location
        :param float width: gaussian sigma
        :return: value
        :rtype: float
        """
        return np.exp(-np.power(x - center, 2.) / (2. * np.power(width, 2.))) * height + baseline

    @staticmethod
    def fit_gaussian_to_histogram(x, y):
        """Finds best fit parameters of :py:func:`UnitDuplicateDetector.histogram_gaussian`
        to data provided.

        Initialization parameters are:
        `baseline = np.median(y)`
        `height = np.max(y) - np.median(y)`
        `center = x[np.argmax(y)]`
        `width = np.array((x[-1] - x[0]) / 2.)`

        :param numpy.ndarray x: histogram bin centers
        :param numpy.ndarray y: histogram bin values
        :return: fitted parameter array [baseline, height, center, width]
        :rtype: numpy.ndarray
        """
        initial_params = (np.median(y), np.max(y) - np.median(y), x[np.argmax(y)], np.array((x[-1] - x[0]) / 2.))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Covariance of the parameters could not be estimated',
                                    category=OptimizeWarning)
            try:
                return curve_fit(UnitDuplicateDetector.histogram_gaussian,
                                 np.array(x), np.array(y), p0=initial_params)[0]
            except RuntimeError:
                return np.array([np.nan, np.nan, np.nan, np.nan])

    def compute_coincident_spike_counts_for_all_unit_pairs(self, max_lag, bin_size, category=None):
        """Computes coincident spike counts, fits gaussians to resulting histograms and stores
        the output to recording.analysis['unit_coincident_spikes'] of the first recording.

        :param float max_lag: maximum time lag to query
        :param float bin_size: bin size of time lags
        :param str category: if provided, both units must belong to this category to be included
        """

        unit_pairs = deepcopy(self.unit_pairs)

        # Filter by unit category if specified
        if not (category is None):
            unit_pairs = filter(lambda x: self.get_unit_category(x[0]) == self.get_unit_category(x[1]) == category,
                                unit_pairs)

        # Prepare data for calculations
        bin_centers = self.bin_centers(max_lag, bin_size)
        unit_pairs = sorted(unit_pairs)

        if self._verbose:
            print('Computing coincident spike counts for recording(s).')

        # Compute coincident spike counts and fit gaussians to resulting histograms
        counts = []
        fit_params = []
        for (i, j) in (tqdm(unit_pairs) if self._verbose else unit_pairs):

            counts.append(self.compute_coincident_spike_counts(i, j, max_lag, bin_size))
            fit_params.append(self.fit_gaussian_to_histogram(bin_centers, counts[-1]))

        if isinstance(self._recordings, Recording):
            self._recordings.analysis['unit_coincident_spikes'] = {'counts': np.array(counts),
                                                                   'fit_params': np.array(fit_params),
                                                                   'unit_pairs': np.array(unit_pairs)}
        elif isinstance(self._recordings, Recordings):
            self._recordings[0].analysis['unit_coincident_spikes'] = {'counts': np.array(counts),
                                                                      'fit_params': np.array(fit_params),
                                                                      'unit_pairs': np.array(unit_pairs)}

    def get_unit_ratemaps(self, i, ratemap_min_spikes=0):
        """Returns list of ratemaps for unit. If unit is None for one of the recordings,
        the returned list will contain a None in-place of the ratemap for that recording.

        :param int i: unit index
        :param int ratemap_min_spikes: minimum number of spikes in recording corresponding to the ratemap,
            otherwise None is returned in place of the ratemap.
        :return: list of ratemaps
        :rtype: list
        """
        if isinstance(self._recordings, Recording):
            if len(self._recordings.units[i]['timestamps']) < ratemap_min_spikes:
                return [None]
            else:
                return [self._recordings.units[i]['analysis']['spatial_ratemaps']['spike_rates_smoothed']]
        elif isinstance(self._recordings, Recordings):
            return [self._recordings.units[i][n]['analysis']['spatial_ratemaps']['spike_rates_smoothed']
                    if (not (self._recordings.units[i][n] is None)
                        and len(self._recordings.units[i][n]['timestamps']) >= ratemap_min_spikes)
                    else None
                    for n in range(len(self._recordings))]

    def compute_ratemap_correlation_for_units(self, i, j, ratemap_min_spikes=0):
        """Returns spatial ratemaps corelations for all recordings where both units are active.

        :param int i: first unit index
        :param int j: second unit index
        :param int ratemap_min_spikes: minimum number of spikes required in the ratemap corresponding to recording,
            otherwise the ratemap correlation is skipped and output list has one less element.
        :return: correlations
        :rtype: list
        """
        correlations = []
        for ratemap_i, ratemap_j in zip(self.get_unit_ratemaps(i, ratemap_min_spikes=ratemap_min_spikes),
                                        self.get_unit_ratemaps(j, ratemap_min_spikes=ratemap_min_spikes)):
            if not (ratemap_i is None) and not (ratemap_j is None):
                correlations.append(spatial_correlation(ratemap_i, ratemap_j)[0])

        return correlations

    def list_duplicates(self, max_lag, bin_size, coincident_spikes_peak_width_threshold,
                        coincidental_spike_count_threshold, ratemap_correlation_threshold,
                        ratemap_min_spikes=0, category=None):
        """Returns a list of unit pairs that are classified as duplicates.

        :param float max_lag: maximum time lag to query for coincident spike histograms
        :param float bin_size: bin size of time lags for coincident spike histograms
        :param float coincident_spikes_peak_width_threshold: maximum peak width for unit pair to be flagged (seconds)
        :param float coincidental_spike_count_threshold: minimum coincident spike count at 0 bin for
            unit pair to be flagged
        :param float ratemap_correlation_threshold: minimum mean ratemap correlation for unit to be flagged
        :param int ratemap_min_spikes: minimum number of spikes required in the recording corresponding to ratemap,
            for the ratemaps to be counted in when filtering by ratemap_correlation_threshold.
        :param category:
        :return: unit_pairs, list of 2-element-tuples for each unit pair
        :rtype: list
        """
        # Get unit_coincident_spikes data
        if isinstance(self._recordings, Recording):
            analysis_dict = self._recordings.analysis
        elif isinstance(self._recordings, Recordings):
            analysis_dict = self._recordings[0].analysis
        else:
            raise Exception('Input to UnitDuplicateDetector is incorrect.')
        if 'unit_coincident_spikes' not in analysis_dict:
            self.compute_coincident_spike_counts_for_all_unit_pairs(max_lag, bin_size, category=category)
        unit_coincident_spikes = analysis_dict['unit_coincident_spikes']

        # Find violations of coincident_spikes_peak_width_threshold and coincidental_spike_count_threshold
        central_bin_ind = unit_coincident_spikes['counts'].shape[1] // 2
        idx = np.logical_and(
            (unit_coincident_spikes['fit_params'][:, 3] <= coincident_spikes_peak_width_threshold),
            (unit_coincident_spikes['counts'][:, central_bin_ind] >= coincidental_spike_count_threshold)
        )

        # Filter these unit pairs by ratemap_correlation_threshold
        for n in np.where(idx)[0]:
            correlations = self.compute_ratemap_correlation_for_units(*unit_coincident_spikes['unit_pairs'][n, :],
                                                                      ratemap_min_spikes=ratemap_min_spikes)
            idx[n] = np.nanmean(np.array(correlations)) >= ratemap_correlation_threshold

        return list(map(tuple, list(unit_coincident_spikes['unit_pairs'][idx, :])))
