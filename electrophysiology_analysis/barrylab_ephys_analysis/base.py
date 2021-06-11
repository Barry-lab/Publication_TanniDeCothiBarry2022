
import numpy as np

from barrylab_ephys_analysis.recording_io import check_if_analysis_field_in_file


class AnalogSignal(object):
    """
    Base class for AnalogSignal, e.g. FrequencyBandPhase.

    If data with matching name is found in recording.analysis['analog_signals'],
    this is used instead of computing, if stored parameters match input arguments.
    """

    def __init__(self, recording, name, *args, **kwargs):
        """
        :param recording: :py:class:`recording_io.Recordings`
        :param str name: will be the :py:attr:`AnalogSignal.name`
        """

        self.recording = recording
        self.name = name

        if 'analog_signals' in self.recording.analysis and self.name in self.recording.analysis['analog_signals']:
            self.data = self.recording.analysis['analog_signals'][self.name]
            print('{} fround in recording.analysis, using as potentially matching data'.format(self.name))
        else:
            self.data = None

        self._timestamps = None

        self._nanmean = None
        self._nanstd = None

    @property
    def nanmean(self):
        if self._nanmean is None:
            self._nanmean = np.nanmean(self.data['signal'], axis=0)[None, :]
        return self._nanmean

    @property
    def nanstd(self):
        if self._nanstd is None:
            self._nanstd = np.nanstd(self.data['signal'], axis=0)[None, :]
        return self._nanstd

    def available_to_load(self):
        AnalogSignal.check_if_saved(self.recording.fpath, self.name)

    def add_to_recording(self):
        if self.data is None:
            raise Exception('data is not yet computed. Can not add to recording.')
        if not ('analog_signals' in self.recording.analysis):
            self.recording.analysis['analog_signals'] = {}
        self.recording.analysis['analog_signals'][self.name] = self.data

    @staticmethod
    def check_if_saved(fpath, name):
        return check_if_analysis_field_in_file(fpath, 'analog_signals/' + name)

    def get_values_between_timestamps(self, start_time, end_time, extend=0, allow_clipping=False, zscored=False):
        """Returns a slice through self.data['signal'] in axis=0 that falls within
        the start and end timestamps provided.

        Additional timesteps before and after the requested range are returned
        for number of time steps specified by extend argument. This can be useful
        when extracting data for later interpolation.

        :param float start_time: signal slice start time
        :param float end_time: signal slice end time
        :param int extend: number of additional datapoints to retrieve before and after. Default is 0.
        :param bool allow_clipping: if False (default), ValueError is raised if time range is outside signal range.
                                    if True, time range is clipped to signal range if necessary.
        :param bool zscored: if True, output is zscored based on full signal (Default is False)
        :return: signal, timestamps
        """
        start_time = float(start_time)
        end_time = float(end_time)

        time_step = 1. / float(self.data['sampling_rate'])

        start_ind = int(np.ceil((start_time - self.data['first_timestamp']) / time_step))
        end_ind = int(np.floor((end_time - self.data['first_timestamp']) / time_step))

        start_ind -= extend
        end_ind += extend

        if start_ind < 0 or end_ind > self.data['signal'].shape[0]:
            if allow_clipping:
                start_ind = 0 if start_ind < 0 else start_ind
                end_ind = self.data['signal'].shape[0] if end_ind > self.data['signal'].shape[0] else end_ind
            else:
                raise ValueError('Requested slice {} : {} is outside the range of signal {}.'.format(
                    start_ind, end_ind, self.name))

        # Take slice of self.data['signal']
        signal = self.data['signal'][start_ind:end_ind, ...].copy()

        if zscored:
            signal = signal - self.nanmean
            signal = signal / self.nanstd

        # Compute timestamps corresponding to elements of this slice
        n_steps = end_ind - start_ind
        timestamps = (np.arange(0, (time_step - np.finfo(type(time_step)).eps) * n_steps, time_step)
                      + (time_step * start_ind + self.data['first_timestamp']))

        return signal, timestamps

    def get_values_interpolated_to_timestamps(self, timestamps, zscored=False):
        """Returns values of self.data['signal'] precisely at requested timestamps through interpolation.

        This method is optimised for requesting values at adjacent timesteps.

        Note, this method only works for signal data with ndim == 2.

        :param numpy.ndarray timestamps: shape (n_samples,) specifying timestamps requested
        :return: interpolated_signal shape (n_samples, self.data['signal'].shape[1])
        """
        signal, original_timestamps = self.get_values_between_timestamps(
            timestamps[0], timestamps[-1], extend=1, zscored=zscored)

        interpolated_signal = np.zeros((len(timestamps), signal.shape[1]), dtype=signal.dtype)

        for n_col in range(signal.shape[1]):
            interpolated_signal[:, n_col] = np.interp(timestamps, original_timestamps, signal[:, n_col])

        return interpolated_signal

    def get_values_of_samples_closest_matching_to_timestamps(self, timestamps):
        """Returns samples of self.data['signal'] for which self.data['timestamps'] are closest to queried timestamps.

        :param numpy.ndarray timestamps: query timepoints to match samples to.
            Note! The timestamps must not be greater or smaller than the smallest first and last timestamp
            of the signal by more than one sample.
        :return: signal_samples
        :rtype: numpy.ndarray
        """
        if (np.min(timestamps) < self.timestamps[0] - 1 / float(self.data['sampling_rate'])
                or np.max(timestamps) > self.timestamps[-1] + 1 / float(self.data['sampling_rate'])):
            raise ValueError('timestamps must not include values more than one sample before or after signal samples.')
        idx = np.searchsorted(self.timestamps, timestamps)
        idx[self.timestamps[idx] - timestamps > (self.timestamps[1] - self.timestamps[0]) * 0.5] -= 1
        return self.data['signal'][idx, ...]

    def get_epoch_timestamps_function_of_signal_true(self, func):
        """Returns a list of epoch timestamps (start_time, end_time) where func(data['signal']) has true values.

        :param func: function that must take one argument AnalogSignal.data['signal'] (n_samples, n_channels)
            and return boolean numpy.ndarray shape (n_samples,)
        :return: epoch_timestamps, list of tuples (start_time, end_time)
        :rtype: list
        """

        idx = func(self.data['signal'])
        d_idx = np.diff(idx.astype(np.int8))

        # Correct d_idx length by adding epoch start signal to beginning if first idx True, else add 0 (no signal)
        if idx[0]:
            d_idx = np.concatenate((np.array([1]), d_idx))
        else:
            d_idx = np.concatenate((np.array([0]), d_idx))
        # If last idx True, ensure last element signals end of epoch if previously True, otherwise ignore element.
        if idx[-1]:
            if idx[-2]:
                d_idx[-1] = -1
            else:
                d_idx[-1] = 0

        # Iterate over all d_idx elements equal to 1 and find next -1 element
        epoch_timestamps = []
        for i in np.where(d_idx == 1)[0]:
            j = i.copy()
            for j in range(i, d_idx.size - 1):
                if d_idx[j] == -1:
                    break
            # Append timestamps corresponding to these elements to epoch_timestamps
            epoch_timestamps.append((self.timestamps[i], self.timestamps[j]))

        return epoch_timestamps

    def compute(self):
        """
        This method should be overwritten for a sub-class.
        The method should store output to `self.data` as a dictionary

        The output must satisfy the requirements to be stored in NWB file
        using :py:func:`NWBio.recursively_save_dict_contents_to_group`.

        The minimum requirements for fields in output dictionary is the following:
        {
            'first_timestamp': numpy.array(timestamp float),
            'sampling_rate': numpy.array(sampling rate float),
            'signal': numpy.array with shape (n_samples, n_channels),
            'channel_labels': list of strings of same length as n_channels
        }
        """
        raise NotImplementedError('This should be overwritten by sub-class to write to self.data')

    @property
    def timestamps(self):
        if self._timestamps is None and not (self.data is None):
            timestep = 1. / float(self.data['sampling_rate'])
            timestamps = np.arange(
                0, self.data['signal'].shape[0] * timestep, timestep)
            self._timestamps = timestamps + self.data['first_timestamp']
        return self._timestamps

    @classmethod
    def get_or_create_instance_for_recording(cls, recording, name, **kwargs):
        """Checks if class instance already exists with `name` in `recording.analog_signals`.
        If not, class is instantiated, computed and added to the recording.

        :param recording: :py:class:`recording_io.Recordings`
        :param str name: will be the :py:attr:`AnalogSignal.name`
        :return: instance of the class
        """

        if hasattr(recording, 'analog_signals'):
            if name in [x.name for x in recording.analog_signals]:
                return [x for x in recording.analog_signals if x.name == name][0]

        print('{} not available, computing and adding to recording ...'.format(name))
        analog_signal = cls(recording, name, **kwargs)
        analog_signal.compute()
        analog_signal.add_to_recording()

        if not hasattr(recording, 'analog_signals'):
            recording.analog_signals = []
        recording.analog_signals.append(analog_signal)

        return analog_signal


class Events(object):

    def __init__(self, recording, name, *args, **kwargs):

        self.recording = recording
        self.name = name

        if 'events' in self.recording.analysis and self.name in self.recording.analysis['events']:
            self.data = self.recording.analysis['events'][self.name]
        else:
            self.data = None

    def available_to_load(self):
        AnalogSignal.check_if_saved(self.recording.fpath, self.name)

    def add_to_recording(self):
        if self.data is None:
            raise Exception('data is not yet computed. Can not add to recording.')
        if not ('events' in self.recording.analysis):
            self.recording.analysis['events'] = {}
        self.recording.analysis['events'][self.name] = self.data

    @staticmethod
    def check_if_saved(fpath, name):
        return check_if_analysis_field_in_file(fpath, 'events/' + name)

    def compute(self):
        """
        This method should be overwritten for a sub-class.
        The method should store output to `self.data` as a dictionary

        The output must satisfy the requirements to be stored in NWB file
        using :py:func:`NWBio.recursively_save_dict_contents_to_group`.

        The minimum requirements for fields in output dictionary is the following:
        {
            'timestamps': numpy.array(timestamp float)
        }
        """
        raise NotImplementedError('This should be overwritten by sub-class to write to self.data')

    @classmethod
    def get_or_create_instance_for_recording(cls, recording, name, **kwargs):
        """Checks if class instance already exists with `name` in `recording.events`.
        If not, class is instantiated, computed and added to the recording.

        :param recording: :py:class:`recording_io.Recordings`
        :param str name: will be the :py:attr:`Events.name`
        :return: instance of the class
        """

        if hasattr(recording, 'events'):
            if name in [x.name for x in recording.events]:
                return [x for x in recording.events if x.name == name][0]

        print('{} not available, computing and adding to recording ...'.format(name))
        events = cls(recording, name, **kwargs)
        events.compute()
        events.add_to_recording()

        if not hasattr(recording, 'events'):
            recording.events = []
        recording.events.append(events)

        return events


class Epochs(object):

    def __init__(self, recording, name, *args, **kwargs):

        self.recording = recording
        self.name = name

        if 'epochs' in self.recording.analysis and self.name in self.recording.analysis['epochs']:
            self.data = self.recording.analysis['epochs'][self.name]
        else:
            self.data = None

    def available_to_load(self):
        AnalogSignal.check_if_saved(self.recording.fpath, self.name)

    def add_to_recording(self):
        if self.data is None:
            raise Exception('data is not yet computed. Can not add to recording.')
        if not ('epochs' in self.recording.analysis):
            self.recording.analysis['epochs'] = {}
        self.recording.analysis['epochs'][self.name] = self.data

    @staticmethod
    def check_if_saved(fpath, name):
        return check_if_analysis_field_in_file(fpath, 'epochs/' + name)

    def find_epochs_overlapping_timewindow(self, timewindow_start, timewindow_end):
        """Returns range of epoch indices that overlap with the period defined by timestamps.

        :param float timewindow_start: start of time window
        :param float timewindow_end: end of time window
        :return: range of epoch indices
        :rtype: range
        """
        # Get epoch timestamps as numpy.array
        start_times = np.array([x[0] for x in self.data['timestamps']])
        end_times = np.array([x[1] for x in self.data['timestamps']])

        # Find potential epoch indices that overlap with input timestamps range
        return np.where(np.logical_or(
            np.logical_and(start_times > timewindow_start, start_times < timewindow_end),
            np.logical_and(end_times > timewindow_start, end_times < timewindow_end)
        ))[0]

    def get_epoch_boolean_at_timestamps(self, timestamps):
        """Returns boolean array same shape as input with True values where epoch is active.

        This method is optimised for requesting values at adjacent timesteps.

        :param numpy.ndarray timestamps: shape (n_samples,) specifying timestamps requested
        :return: boolean array same shape as timestamps with True values where epoch is active
        """

        timestamps = np.array(timestamps)

        epoch_index_range = self.find_epochs_overlapping_timewindow(timestamps[0], timestamps[-1])

        if len(epoch_index_range) > 0:

            first_epoch, last_epoch = (epoch_index_range[0], epoch_index_range[-1])

            # Create epoch_active array and set elements True where within active epochs
            epoch_active = np.zeros(timestamps.shape, dtype=np.bool)
            for epoch_start, epoch_end in self.data['timestamps'][first_epoch:last_epoch + 1]:
                idx = np.logical_and(timestamps < epoch_end, timestamps > epoch_start)
                epoch_active[idx] = True

        else:

            epoch_active = np.zeros(timestamps.shape, dtype=np.bool)

        return epoch_active

    def compute(self):
        """
        This method should be overwritten for a sub-class.
        The method should store output to `self.data` as a dictionary

        The output must satisfy the requirements to be stored in NWB file
        using :py:func:`NWBio.recursively_save_dict_contents_to_group`.

        The minimum requirements for fields in output dictionary is the following:
        {
            'timestamps': numpy.array([[first_epoch_start, first_epoch_end],
                                       [second_epoch_start, second_epoch_end],
                                       ...
                                       ])
        }

        Note, timestamps are assumed to be sorted in ascending order.
        """
        raise NotImplementedError('This should be overwritten by sub-class to write to self.data')

    @classmethod
    def get_or_create_instance_for_recording(cls, recording, name, **kwargs):
        """Checks if class instance already exists with `name` in `recording.epochs`.
        If not, class is instantiated, computed and added to the recording.

        :param recording: :py:class:`recording_io.Recordings`
        :param str name: will be the :py:attr:`Epochs.name`
        :return: instance of the class
        """

        epochs = None

        if hasattr(recording, 'epochs'):
            if name in [x.name for x in recording.epochs]:
                print('{} found, adding this and recomputing (if necessary) ...'.format(name))
                epochs = [x for x in recording.epochs if x.name == name][0]

        if epochs is None:
            print('{} not available, creating a new class ...'.format(name))
            epochs = cls(recording, name, **kwargs)

        epochs.compute()
        epochs.add_to_recording()

        if not hasattr(recording, 'epochs'):
            recording.epochs = []
        recording.epochs.append(epochs)

        return epochs
