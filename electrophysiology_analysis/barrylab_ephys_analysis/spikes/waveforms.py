import numpy as np
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm


def check_if_dtype_suitable(dtype, min_value, max_value):
    return np.iinfo(dtype).max > max_value and np.iinfo(dtype).min < min_value


def find_correct_dtype(original_dtype, min_value, max_value):
    dtype_found = False
    if np.issubdtype(original_dtype, np.integer):
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if check_if_dtype_suitable(dtype, min_value, max_value):
                return dtype
        if not dtype_found:
            raise Exception('Too large value range requested')
    else:
        raise ValueError('original_dtype {} not recoqnised'.format(original_dtype))


class WaveformProperties(object):
    """
    Provides access to common waveform properties
    """

    def __init__(self, waveforms, sampling_rate=30000, gain=0.195):
        """Provides access to properties of waveforms.

        WaveformProperties are computed when first accessed.

        :param numpy.ndarray waveforms: shape (n_spike_samples, n_channels, n_waveform_samples)
        :param float gain: for multiplying voltage values
        :param int sampling_rate: sampling rate of waveforms
        """

        self._waveforms = waveforms
        self._gain = float(gain)
        self._sampling_rate = float(sampling_rate)

        self._amplitude = None
        self._peak_index = None
        self._trough_index = None
        self._time_to_peak = None
        self._time_to_trough = None
        self._peak_to_trough = None
        self._trough_ratio = None
        self._half_width = None
        self._pca_components = None

    @property
    def waveforms(self):
        return self._waveforms

    @property
    def gain(self):
        return self._gain

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def all(self):
        return {'amplitude': self.amplitude,
                'time_to_peak': self.time_to_peak,
                'time_to_trough': self.time_to_trough,
                'peak_to_trough': self.peak_to_trough,
                'trough_ratio': self.trough_ratio,
                'half_width': self.half_width,
                'pca_component_1': self.pca_component_1,
                'pca_component_2': self.pca_component_2,
                'pca_component_3': self.pca_component_3}

    @property
    def amplitude(self):
        if self._amplitude is None:
            self._amplitude = np.float32(WaveformProperties.compute_amplitude(self.waveforms, self.gain))
        return self._amplitude

    @property
    def time_to_peak(self):
        if self._time_to_peak is None:
            self._time_to_peak = np.float32(WaveformProperties.compute_time_to_peak(self.peak_index,
                                                                                    self.sampling_rate))
        return self._time_to_peak

    @property
    def time_to_trough(self):
        if self._time_to_trough is None:
            self._time_to_trough = np.float32(WaveformProperties.compute_time_to_trough(self.trough_index,
                                                                                        self.sampling_rate))
        return self._time_to_trough

    @property
    def peak_to_trough(self):
        if self._peak_to_trough is None:
            self._peak_to_trough = np.float32(WaveformProperties.compute_peak_to_trough(self.time_to_peak,
                                                                                        self.time_to_trough))
        return self._peak_to_trough

    @property
    def trough_ratio(self):
        if self._trough_ratio is None:
            self._trough_ratio = np.float32(WaveformProperties.compute_trough_ratio(self.waveforms, self.gain,
                                                                                    self.amplitude,
                                                                                    self.trough_index))
        return self._trough_ratio

    @property
    def half_width(self):
        if self._half_width is None:
            self._half_width = np.float32(WaveformProperties.compute_half_width(self.waveforms,
                                                                                self.sampling_rate))
        return self._half_width

    @property
    def pca_components(self):
        if self._pca_components is None:
            self._pca_components = WaveformProperties.compute_pca_components(
                self.waveforms, n_components=3, energy_normalised=True)
        return self._pca_components

    @property
    def pca_component_1(self):
        return self.pca_components[:, :, 0].squeeze()

    @property
    def pca_component_2(self):
        return self.pca_components[:, :, 1].squeeze()

    @property
    def pca_component_3(self):
        return self.pca_components[:, :, 2].squeeze()

    @property
    def peak_index(self):
        if self._peak_index is None:
            self._peak_index = WaveformProperties.compute_peak_index(self.waveforms)
        return self._peak_index

    @property
    def trough_index(self):
        if self._trough_index is None:
            self._trough_index = WaveformProperties.compute_trough_index(self.waveforms)
        return self._trough_index

    @staticmethod
    def compute_amplitude(waveforms, gain):
        return np.ptp(waveforms, axis=-1).astype(np.float32) * float(gain)

    @staticmethod
    def compute_peak_index(waveforms):
        return np.argmin(waveforms, axis=-1)

    @staticmethod
    def compute_time_to_peak(peak_index, sampling_rate):
        return peak_index / float(sampling_rate)

    @staticmethod
    def compute_trough_index(waveforms):
        return np.apply_along_axis(lambda x: np.argmin(x) + np.argmax(x[np.argmin(x):]), -1, waveforms)

    @staticmethod
    def compute_time_to_trough(trough_index, sampling_rate):
        return trough_index / float(sampling_rate)

    @staticmethod
    def compute_peak_to_trough(time_to_peak, time_to_trough):
        return time_to_trough - time_to_peak

    @staticmethod
    def compute_trough_ratio(waveforms, gain, amplitude, trough_index):
        return (np.take_along_axis(waveforms, trough_index[:, :, np.newaxis], axis=2).squeeze().astype(np.float32)
                * gain / amplitude)

    @staticmethod
    def compute_waveform_half_width(waveform):
        # Compute a single waveform half_width in number of samples
        half_width_height = (waveform[0] + np.min(waveform)) / 2.0
        first_crossing = np.argmax(waveform < half_width_height)
        second_crossing = np.argmax(waveform[first_crossing:] > half_width_height) + first_crossing
        return second_crossing - first_crossing

    @staticmethod
    def compute_half_width(waveforms, sampling_rate):
        return np.apply_along_axis(WaveformProperties.compute_waveform_half_width, -1, waveforms) / float(sampling_rate)

    @staticmethod
    def energy_normalize_waveforms(waveforms):

        # Find channels with 0 variance (bad channels)
        bad_chan = (np.apply_over_axes(np.var, waveforms, axes=(0, 2)) == 0).squeeze()

        # Ensure dtype allows for values necessary for this computation
        original_dtype = waveforms.dtype
        if not check_if_dtype_suitable(
                original_dtype, 0, np.amax(np.abs(waveforms)).astype(np.int64) ** 2):
            waveforms = waveforms.astype(
                find_correct_dtype(
                    original_dtype, 0, np.amax(np.abs(waveforms)).astype(np.int64) ** 2
                )
            )

        # Compute energy of all waveforms
        energy = np.sqrt(np.nansum(waveforms ** 2, axis=2))

        # Put waveforms and energy to float format as normalised values will be below 0
        waveforms = np.float32(waveforms)
        energy = np.float32(energy)

        # Normalise waveform values with energy, ignoring bad channels
        energy = np.stack([energy] * waveforms.shape[2], axis=2)
        waveforms[:, ~bad_chan, :] = waveforms[:, ~bad_chan, :] / energy[:, ~bad_chan, :]

        return waveforms

    @staticmethod
    def compute_pca_components(waveforms, n_components=3, energy_normalised=True):

        if energy_normalised:
            waveforms = WaveformProperties.energy_normalize_waveforms(waveforms)

        with warnings.catch_warnings():
            # Ignorning warnigs of arrays with 0 values
            warnings.filterwarnings(action='ignore',
                                    message='invalid value encountered in true_divide',
                                    category=RuntimeWarning)
            if waveforms.ndim == 2:
                return PCA(n_components=n_components).fit_transform(waveforms)
            elif waveforms.ndim == 3:
                components = np.zeros((waveforms.shape[0], waveforms.shape[1], n_components), dtype=np.float32)
                for i in range(waveforms.shape[1]):
                    components[:, i, :] = PCA(n_components=n_components).fit_transform(waveforms[:, i, :])
                return components

    @staticmethod
    def add_waveform_properties_to_recording_analysis(recording):
        """Computes all waveform properties and adds them to unit analysis field in recording.

        :param recording_io.Recording recording: `recording_io.Recording` class instance.
        """
        for unit in tqdm(recording.units):
            unit['analysis']['waveform_properties'] = WaveformProperties(
                unit['waveforms'], unit['sampling_rate'], recording.microvolt_gain).all

    @staticmethod
    def check_if_available_in_recording_else_compute(recording):
        """Checks if a unit has waveform_properties field in analysis dictionary.
        If not, computes the properties using `WaveformProperties.all`

        :param recording_io.Recording recording: `recording_io.Recording` class instance.
        """
        for unit in tqdm(recording.units):
            if not('waveform_properties' in unit['analysis']):
                unit['analysis']['waveform_properties'] = WaveformProperties(
                    unit['waveforms'], unit['sampling_rate'], recording.microvolt_gain).all


class RecordingUnitListWaveformProperties(object):
    """
    Class for using :py:class:`WaveformProperties` concurrently for waveforms
    of multiple units.

    This is useful in at least two cases:
    * Computing principal components (PCA) concurrently for waveforms of all units
    recorded on a single tetrode or other channel group.
    * Computing PCA and mean waveforms for units across multiple recordings

    Assumes waveforms in all units have the same gain and sampling rate.

    """

    def __init__(self, units, sampling_rate, gain):
        """Instantiates :py:class`WaveformProperties` for all waveforms of all units

        :param list units: list of units as in :py:attr:`recording_io.Recording.units`
            Can be list of units combined across multiple recordings as with
            :py:func:`recordings.get_units_for_one_tetrode_as_one_level_list`
        :param sampling_rate:
        :param gain:
        """

        # Count number of spikes per unit
        self.n_spikes = []
        for unit in units:
            self.n_spikes.append(unit['timestamps'].size if not (unit is None) else 0)

        # Instantiate WaveformProperties with concatenated waveform array
        self.waveform_properties = WaveformProperties(
            np.concatenate([unit['waveforms'] for unit in units if not (unit is None)],
                           axis=0),
            sampling_rate,
            gain
        )

    def get_waveform_property(self, unit_index, property_name):
        """Returns the specified property from WaveformProperties slicing
        only for the spikes specific to specified unit.

        :param int unit_index: position of unit in units list provided during initiation
        :param str property_name: name of the property to get from WaveformProperties
        """
        start_index = sum(self.n_spikes[:unit_index])
        end_index = start_index + self.n_spikes[unit_index]

        if property_name == 'all':
            return {key: value[start_index:end_index, ...]
                    for key, value in self.waveform_properties.all.items()}
        else:
            return getattr(self.waveform_properties, property_name)[start_index:end_index, ...]

    @staticmethod
    def store_all_waveform_properties_for_units_in_list_in_analysis(units, sampling_rate, gain):
        """Computes all waveform properties concurrently for units provided and stores them
        in unit['analysis']['waveform_properties'] for each unit.

        Units should either be a list of unit dictionaries as :py:attr:`recording_io.Recording.units`
        or :py:func:`recording_io.Recordings.get_units_for_one_tetrode_as_one_level_list`.

        Usually this method would be used on a list of units detected on the same tetrode or
        other proximal channel group.

        :Example:

        >>> from barrylab_ephys_analysis.recording_io import Recordings
        >>> from barrylab_ephys_analysis.spikes.waveforms import RecordingUnitListWaveformProperties as ReUnLiWaPo
        >>> recordings = Recordings(['/data1/filename', '/data2/filename'])
        >>> tetrode_units, tetrode_unit_group_ids = /
        >>>     recordings.get_units_for_one_tetrode_as_one_level_list(3)
        >>> ReUnLiWaPo.store_all_waveform_properties_for_units_in_list_in_analysis(
        >>>     tetrode_units, recordings.waveform_sampling_rate, recordings.microvolt_gain)

        :param list units: list of unit dictionaries
        :param int sampling_rate: sampling rate of waveforms Hz
        :param float gain: multiplier for converting waveform values to microvolts
        """
        unit_list_properties = RecordingUnitListWaveformProperties(units, sampling_rate, gain)
        for i, unit in enumerate(units):
            if unit is None:
                continue
            unit['analysis']['waveform_properties'] = unit_list_properties.get_waveform_property(i, 'all')


def mean_waveform(waveforms, gain):
    """Returns the mean waveform shape.

    :param numpy.ndarray waveforms: shape (n_spikes, n_channels, n_timesamples)
    :param float gain: waveforms are multiplied with this value
    :return: shape (n_channels, n_timesamples)
    :rtype: numpy.ndarray
    """
    return np.mean(waveforms, axis=0).squeeze() * gain


def store_waveform_shapes_in_analysis(unit, gain):
    """Computes unit waveform shape and stores it in `unit['analysis']['mean_waveform']`.

    :param dict unit: :py:class:`Recording` attribute `unit` element.
    :param float gain: waveforms are multiplied with this value
    """
    unit['analysis']['mean_waveform'] = mean_waveform(unit['waveforms'], gain)


def store_mean_waveform_properties_in_analysis(unit):
    """Creates unit['analysis']['mean_waveform_properties'] dictionary which contains
     numpy.nanmean(x, axis=0) for each element in unit['analysis']['waveform_properties'].

    :param dict unit: :py:class:`Recording` attribute `unit` element.
    """
    if 'analysis' in unit and 'waveform_properties' in unit['analysis']:
        unit['analysis']['mean_waveform_properties'] = {}
        for key, item in unit['analysis']['waveform_properties'].items():
            unit['analysis']['mean_waveform_properties'][key] = np.nanmean(item, axis=0)
    else:
        raise Exception('Input unit dictionary does not contain unit["analysis"]["waveform_properties"]')


def compute_all_waveform_properties_for_all_units_in_recordings(recordings, verbose=False):
    """Computes all waveform properties of all tetrodes (including mean_waveform and
    mean_waveform_properties) and stores them in `analysis` field of each `unit` dictionary.

    :param recordings: :py:class:`recording_io.Recordings` instance
    :param bool verbose: if True, prints progress information. Default is False.
    """
    # Loop through all tetrodes
    if verbose:
        print('Computing waveform_properties for {} {}'.format(recordings.info[0]['animal'],
                                                               recordings.info[0]['rec_datetime']))
    iterable = recordings.get_list_of_tetrode_nrs_across_recordings()
    for tetrode_nr in (tqdm(iterable) if verbose else iterable):

        # Get units on this tetrode across all recordings
        unit_list, _ = recordings.get_units_for_one_tetrode_as_one_level_list(tetrode_nr)

        # Compute mean_waveform for each unit in the list
        for unit in unit_list:
            if not (unit is None):
                store_waveform_shapes_in_analysis(unit, recordings.microvolt_gain)

        # Compute waveform properties concurrently for all units in this list
        RecordingUnitListWaveformProperties.store_all_waveform_properties_for_units_in_list_in_analysis(
            unit_list, recordings.waveform_sampling_rate, recordings.microvolt_gain
        )

        # Compute mean_waveform properties for each unit in the list
        for unit in unit_list:
            if not (unit is None):
                store_mean_waveform_properties_in_analysis(unit)
