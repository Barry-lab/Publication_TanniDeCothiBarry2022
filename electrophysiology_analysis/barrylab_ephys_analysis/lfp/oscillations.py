
import numpy as np

from barrylab_ephys_analysis.blea_utils import smooth_signal_with_gaussian
from barrylab_ephys_analysis.lfp.processing import bandpass_filter, hilbert_fast
from barrylab_ephys_analysis.base import AnalogSignal


class FrequencyBandAmplitude(AnalogSignal):
    """
    Computes frequency band amplitude for a recording and provides convenient access based on
    :py:class:`barrylab_ephys_analysis.base.AnalogSignal`.

    If existing data with same name is found in recording.analysis['analog_signals'],
    then that data is used instead, if it matches the input parameters.
    """

    def __init__(self, *args, highpass_frequency=None, lowpass_frequency=None,
                 filter_order=None, temporal_smoothing_sigma=None, common_average=False,
                 channel_group_tetrodes=None, verbose=False, **kwargs):
        """
        :param highpass_frequency: Hz (must be provided)
        :param lowpass_frequency: Hz (must be provided)
        :param int filter_order: (must be provided)
        :param float temporal_smoothing_sigma: temporal smoothing gaussian stdev in seconds (must be provided)
        :param bool common_average: specifies if common average is compute for each channel group before processing.
            Default is False, no common averaging.
        :param dict channel_group_tetrodes: {channel_group_A: 5, channel_group_B: [6, 8]} specifies that for
            channel_group_A, only tetrode 5 is used and for channel_group_B, tetrodes 6 and 8 are used.
            Default is to use all tetrodes on all channel groups. This must be provided for all groups or none.
        :param bool verbose: if True, prints progress. Default is False.
        """
        self._params = {
            'highpass_frequency': np.array(highpass_frequency),
            'lowpass_frequency': np.array(lowpass_frequency),
            'filter_order': np.array(filter_order),
            'temporal_smoothing_sigma': np.array(temporal_smoothing_sigma),
            'common_average': np.array(common_average),
            'channel_group_tetrodes': ({key: np.array(value) for key, value in channel_group_tetrodes.items()}
                                       if isinstance(channel_group_tetrodes, dict) else np.array(np.nan))
        }
        self.verbose = verbose
        super(FrequencyBandAmplitude, self).__init__(*args, **kwargs)

    @staticmethod
    def compute_frequency_band_amplitude(continuous, sampling_rate, highpass_frequency,
                                         lowpass_frequency, filter_order):
        """Returns amplitude envelope of a signal in 1-D array

        :param numpy.ndarray continuous: shape (N,)
        :param sampling_rate: Hz
        :param highpass_frequency: Hz
        :param lowpass_frequency: Hz
        :param int filter_order:
        :return: amplitude_envelope
        :rtype: numpy.ndarray
        """
        filtered_signal = bandpass_filter(continuous, sampling_rate=sampling_rate,
                                          lowpass_frequency=lowpass_frequency,
                                          highpass_frequency=highpass_frequency,
                                          filter_order=filter_order)
        return np.abs(hilbert_fast(filtered_signal))

    def compute_values_for_continuous_array(self, continuous):
        """Computes frequency band amplitude for along first dimension for all columns.

        This can be overwritten by a subclass to provide a different value for continuous array,
        but must return an array of same shape as it receives.

        Note! The output from this function is averaged along the second dimension, using np.mean(x, axis=1).

        :param numpy.ndarray continuous: shape (n_samples, n_channels)
        :return: value_array shape (n_samples, n_channels)
        :rtype: numpy.ndarray
        """
        return np.apply_along_axis(
            FrequencyBandAmplitude.compute_frequency_band_amplitude, 0, continuous,
            self.recording.continuous['sampling_rate'],
            highpass_frequency=self._params['highpass_frequency'],
            lowpass_frequency=self._params['lowpass_frequency'],
            filter_order=self._params['filter_order']
        )

    @staticmethod
    def smooth_signal(signal, temporal_smoothing_sigma, sampling_rate):
        return smooth_signal_with_gaussian(signal, temporal_smoothing_sigma * sampling_rate)

    def compute_for_channel_group(self, channel_group):

        if isinstance(self._params['channel_group_tetrodes'], dict):
            tetrode_nrs = self._params['channel_group_tetrodes'][channel_group]
        else:
            tetrode_nrs = self.recording.channel_map[channel_group]['tetrode_nrs']

        if self._params['common_average'] and len(tetrode_nrs) > 1:

            common_average = np.mean(self.recording.continuous['continuous'][:, tetrode_nrs], axis=1)
            continuous = np.apply_along_axis(lambda x: x - common_average, 0,
                                             self.recording.continuous['continuous'][:, tetrode_nrs])
        else:

            continuous = self.recording.continuous['continuous'][:, tetrode_nrs]

        if len(continuous.shape) == 1:
            continuous = continuous[:, np.newaxis]

        amplitude = np.mean(self.compute_values_for_continuous_array(continuous), axis=1)

        if (not (self._params['temporal_smoothing_sigma'] is None)
                and self._params['temporal_smoothing_sigma'] != 0):
            amplitude = self.smooth_signal(amplitude, self._params['temporal_smoothing_sigma'],
                                           self.recording.continuous['sampling_rate'])

        return amplitude

    def compute(self, recompute=False):

        # If available data params match input params, skip recomputing, unless requested
        if not recompute and not (self.data is None):
            params_match = []
            for key, item in self._params.items():
                if key == 'channel_group_tetrodes' and not isinstance(item, dict):
                    continue
                params_match.append(self.data['params'][key] == item)
            if all(params_match):
                if self.verbose:
                    print('Exisiting {} params match input, skipping computation.'.format(self.name))
                return

        if self.verbose:
            print('Computing {}'.format(self.name))

        signals = []
        labels = []

        for channel_group in self.recording.channel_map:
            signals.append(self.compute_for_channel_group(channel_group)[:, None])
            labels.append(channel_group)

        self.data = {'first_timestamp': np.array(self.recording.continuous['timestamps'][0]),
                     'sampling_rate': np.array(self.recording.continuous['sampling_rate']),
                     'signal': np.concatenate(signals, axis=1),
                     'channel_labels': labels,
                     'params': self._params}


class FrequencyBandFrequency(FrequencyBandAmplitude):
    """
    Computes frequency band instantaneous frequency for a recording and provides convenient access based on
    :py:class:`FrequencyBandAmplitude` and :py:class:`barrylab_ephys_analysis.base.AnalogSignal`.
    """

    @staticmethod
    def compute_frequency_band_frequency(continuous, sampling_rate, highpass_frequency,
                                         lowpass_frequency, filter_order):
        """Returns instantaneous frequency of a signal in 1-D array.

        Note! Due to the nature of computing instantaneous frequency, the values are
        shifted forward by half a sample.

        :param numpy.ndarray continuous: shape (N,)
        :param sampling_rate: Hz
        :param highpass_frequency: Hz
        :param lowpass_frequency: Hz
        :param int filter_order:
        :return: instantaneous_frequency shape (N,)
        :rtype: numpy.ndarray
        """
        filtered_signal = bandpass_filter(continuous, sampling_rate=sampling_rate,
                                          lowpass_frequency=lowpass_frequency,
                                          highpass_frequency=highpass_frequency,
                                          filter_order=filter_order)
        instantaneous_phase = np.unwrap(np.angle(hilbert_fast(filtered_signal)))
        instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * sampling_rate)
        instantaneous_frequency = np.concatenate((instantaneous_frequency[:1], instantaneous_frequency))
        return instantaneous_frequency

    def compute_values_for_continuous_array(self, continuous):
        """Computes instantaneous frequency along first dimension for all columns.

        :param numpy.ndarray continuous: shape (n_samples, n_channels)
        :return: value_array shape (n_samples, n_channels)
        :rtype: numpy.ndarray
        """
        return np.apply_along_axis(
            FrequencyBandFrequency.compute_frequency_band_frequency, 0, continuous,
            self.recording.continuous['sampling_rate'],
            highpass_frequency=self._params['highpass_frequency'],
            lowpass_frequency=self._params['lowpass_frequency'],
            filter_order=self._params['filter_order']
        )


class FrequencyBandPhase(FrequencyBandAmplitude):
    """
    Computes frequency band instantaneous phase for a recording and provides convenient access based on
    :py:class:`FrequencyBandAmplitude` and :py:class:`barrylab_ephys_analysis.base.AnalogSignal`.

    Phase is computed and stored in radians in unwrapped form.

    .. warning::

        If temporal_smoothing_sigma argument above 0 is provided, then phase is unwrapped using
        :py:func:`numpy.unwrap` and wrapped again using :py:func:`FrequencyBandPhase.wrap_phase_to_range_between_pi`.
        It is important that the sampling rate of the signal is sufficiently high for this method to work.

    """

    @staticmethod
    def wrap_phase_to_range_between_pi(signal):
        """Returns input signal wrapped to range between -pi to pi

        :param numpy.ndarray signal:
        :return: wrapped_signal
        """
        return np.mod(signal + np.pi, 2 * np.pi) - np.pi

    @staticmethod
    def compute_frequency_band_phase(continuous, sampling_rate, highpass_frequency,
                                     lowpass_frequency, filter_order):
        """Returns instantaneous phase of a signal in 1-D array in unwrapped form.

        :param numpy.ndarray continuous: shape (N,)
        :param sampling_rate: Hz
        :param highpass_frequency: Hz
        :param lowpass_frequency: Hz
        :param int filter_order:
        :return: instantaneous_phase shape (N,)
        :rtype: numpy.ndarray
        """
        filtered_signal = bandpass_filter(continuous, sampling_rate=sampling_rate,
                                          lowpass_frequency=lowpass_frequency,
                                          highpass_frequency=highpass_frequency,
                                          filter_order=filter_order)
        instantaneous_phase = np.angle(hilbert_fast(filtered_signal))
        return instantaneous_phase

    def compute_values_for_continuous_array(self, continuous):
        """Computes instantaneous phase along first dimension for all columns.

        :param numpy.ndarray continuous: shape (n_samples, n_channels)
        :return: value_array shape (n_samples, n_channels)
        :rtype: numpy.ndarray
        """
        return np.apply_along_axis(
            FrequencyBandPhase.compute_frequency_band_phase, 0, continuous,
            self.recording.continuous['sampling_rate'],
            highpass_frequency=self._params['highpass_frequency'],
            lowpass_frequency=self._params['lowpass_frequency'],
            filter_order=self._params['filter_order']
        )

    @staticmethod
    def smooth_signal(signal, temporal_smoothing_sigma, sampling_rate):
        return FrequencyBandPhase.wrap_phase_to_range_between_pi(
            smooth_signal_with_gaussian(np.unwrap(signal, axis=0), temporal_smoothing_sigma * sampling_rate)
        )


class ThetaDeltaRatio(AnalogSignal):

    def __init__(self, *args, theta_band_amplitude=None, delta_band_amplitude=None,
                 speed=None, mean_ratio_min_speed=None, **kwargs):
        """
        :param theta_band_amplitude: :py:class:`FrequencyBandAmplitude` instantiated for theta frequency band
        :param delta_band_amplitude: :py:class:`FrequencyBandAmplitude` instantiated for delta frequency band
        :param speed: :py:class:`barrylab_ephys_analysis.base.AnalogSignal` for animal speed
        :param float mean_ratio_min_speed: minimum animal speed threshold for calculating mean theta/delta ratio
        """
        self._theta_band_amplitude = theta_band_amplitude
        self._delta_band_amplitude = delta_band_amplitude
        self._speed = speed
        self._params = {'mean_ratio_min_speed': np.array(mean_ratio_min_speed)}
        super(ThetaDeltaRatio, self).__init__(*args, **kwargs)

    def compute(self, recompute=False):

        # If available data params match input params, skip recomputing, unless requested
        if not recompute and not (self.data is None):
            if all([self.data['params'][key] == item for key, item in self._params.items()]):
                return

        print('Computing ThetaDeltaRatio')

        # Compute mean np.log(ratio) of theta/delta amplitude during running epochs
        running_epoch_timestamps = self._speed.get_epoch_timestamps_function_of_signal_true(
            lambda x: np.array(x > self._params['mean_ratio_min_speed']).squeeze()
        )
        running_theta_amplitude = np.mean(
            np.concatenate(
                [self._theta_band_amplitude.get_values_between_timestamps(*ets, allow_clipping=True)[0]
                 for ets in running_epoch_timestamps]
            ),
            axis=1
        )
        running_delta_amplitude = np.mean(
            np.concatenate(
                [self._delta_band_amplitude.get_values_between_timestamps(*ets, allow_clipping=True)[0]
                 for ets in running_epoch_timestamps]
            ),
            axis=1
        )
        running_mean_ratio = np.mean(np.log(running_theta_amplitude / running_delta_amplitude))

        # Get theta/delta ratio with timestamps
        ratio = np.log(np.mean(self._theta_band_amplitude.data['signal'], axis=1)
                       / np.mean(self._delta_band_amplitude.data['signal'], axis=1))

        self.data = {'first_timestamp': self._theta_band_amplitude.data['first_timestamp'],
                     'sampling_rate': self._theta_band_amplitude.data['sampling_rate'],
                     'signal': ratio[:, None],
                     'channel_labels': [''],
                     'params': self._params,
                     'running_mean_ratio': np.array(running_mean_ratio)}
