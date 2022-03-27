# -*- coding: utf-8 -*-

"""
Classes for loading recorded data into convenient format for analysis

"""

import numpy as np
import os
from datetime import datetime
import warnings
from copy import copy, deepcopy
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

from tqdm import tqdm

from barrylab_ephys_analysis.blea_utils import check_if_dictionaries_match
from openEPhys_DACQ import NWBio
from openEPhys_DACQ.HelperFunctions import closest_argmin


def convert_recording_time_string_to_datetime(time_string):
    return datetime.strptime(time_string, '%Y-%m-%d_%H-%M-%S')


def indices_where_tetrode_nrs_in_tetrode_nrs(tetrode_nrs_a, tetrode_nrs_b):
    """Returns indices of tetrode_nrs_b that have values of tetrode_nrs_a in sorted order.

    :param list tetrode_nrs_a:
    :param list tetrode_nrs_b:
    :return: indices in tetrode_nrs_b in sorted order
    :rtype: list
    """
    _, _, ind = np.intersect1d(tetrode_nrs_a, tetrode_nrs_b, assume_unique=True, return_indices=True)
    return list(ind)


def parse_fpath(fpath):
    # Append experiment_1.nwb to path if directory given
    fpath = NWBio.get_filename(fpath)
    if not NWBio.check_if_open_ephys_nwb_file(fpath):
        raise Exception('File is not a recognised format: ' + fpath)

    return fpath


def unit_analysis_record_validation_dict(unit):
    return {'tetrode_nr': np.array(unit['tetrode_nr']),
            'tetrode_cluster_id': np.array(unit['tetrode_cluster_id']),
            'num_clusters': np.array(unit['timestamps'].size)}


def continuous_record_validation_dict(continuous):
    return {'channels': np.array(continuous['channels']),
            'sampling_rate': np.array(continuous['sampling_rate']),
            'first_timestamp': np.array(continuous['timestamps'][0]),
            'last_timestamp': np.array(continuous['timestamps'][-1])}


def position_record_validation_dict(position):
    return {'sampling_rate': np.array(position['sampling_rate']),
            'first_timestamp': np.array(position['timestamps'][0]),
            'last_timestamp': np.array(position['timestamps'][-1])}


def check_if_analysis_field_in_file(fpath, analysis_path):
    fpath = parse_fpath(fpath)
    return NWBio.check_if_path_exists(fpath, '/analysis/' + analysis_path)


class Recording(object):

    def __init__(self, fpath, continuous_data_type='downsampled',
                 spike_name='spikes', clustering_name=None, no_waveforms=False, no_spikes=False,
                 ignore_cluster_zero=False, position_data_name='ProcessedPos', verbose=False):

        """Loads recording data into memory in a convenient format for Neo IO

        Time 0 is at first position data datapoint and all data before the first
        and after the last position data datapoint is discarded.
        If position data is not provided, continuous data datapoints are used as limits instead.

        All timestamps are in seconds.

        All voltage values are in int16 that can be converted to microvolts by multipling with bitVolts (0.195)

        Position data is in centimeters.

        :param str fpath: path to NWB file
        :param continuous_data_type: 'downsampled' or 'raw' to load raw signals or downsampled signals.
                                     if None, no continuous data will be loaded.
        :type continuous_data_type: str or None
        :param str spike_name: spike field to load from NWB file. Default is 'spikes'. See NWBio.load_spikes docs.
        :param clustering_name: allows specifying curated clusterID set to use, see NWBio.load_spikes docs.
                                if None, non-curated clusterIDs will be loaded if available.
        :type clustering_name: str or None
        :param bool no_waveforms: if True, no waveforms are loaded. Default is False.
        :param bool no_spikes: if True, no spikes are loaded. Default is False.
        :param bool ignore_cluster_zero: if True, clusterIDs==0 are ignored. Default is False.
        :param position_data_name: Name of position data to load. See NWBio.load_processed_tracking_data docs.
                                   If None, no position data is loaded.
        :type position_data_name: str or None
        """

        self.verbose = verbose

        self._fpath = parse_fpath(fpath)

        # Get information on this recording
        self._info = Recording.load_file_info(self.fpath)

        # Get channel map for the recording
        self._channel_map = NWBio.get_channel_map_with_tetrode_nrs(self.fpath)

        # Specify first and last timestamp for which to include data
        if position_data_name is None:
            self._first_original_timestamp = NWBio.get_recording_start_timestamp_offset(self.fpath)
            self._last_original_timestamp = self.first_original_timestamp + self.info['continuous_data_duration']
        else:
            self._first_original_timestamp, self._last_original_timestamp = \
                NWBio.get_processed_tracking_data_timestamp_edges(self.fpath)

        # Set Recording instance first and last timestamps
        self._info['first_timestamp'] = 0.0
        self._info['last_timestamp'] = self.last_original_timestamp - self.first_original_timestamp

        # Add cropped data duration to info
        self._info['cropped_duration'] = self._info['last_timestamp'] - self._info['first_timestamp']

        # Set microvolt multiplier
        self._microvolt_gain = NWBio.bitVolts()

        # Set unknown variables
        self._waveform_sampling_rate = None

        # Load continuous data
        self._continuous = None
        self.load_continuous_data(continuous_data_type)

        # Load spike trains
        self._units = []
        self._unit_lookup_table = {}
        if not no_spikes:
            self.load_units(spike_name=spike_name, clustering_name=clustering_name,
                            no_waveforms=no_waveforms, ignore_cluster_zero=ignore_cluster_zero)

        # Load position data
        self._position = None
        self._position_data_sampling_rate = 30
        self.load_position_data(position_data_name=position_data_name,
                                sampling_rate=self.position_data_sampling_rate)

        # Load task data
        self._task_log_parser = None
        self._task_data = {}
        self.load_task_data()

        # Load task settings
        self._task_settings = None
        self.load_task_settings()

        # Create analysis dictionary for appending analysis results to recording
        self._analysis = Recording.__empty_analysis_dictionary()

    @property
    def fpath(self):
        return copy(self._fpath)

    @property
    def info(self):
        return copy(self._info)

    @property
    def identifier(self):
        return self.info['rec_datetime'].__str__()

    @property
    def channel_map(self):
        return copy(self._channel_map)

    @property
    def tetrode_nrs(self):
        """List of all tetrode_nr values present in `Recording.channel_map`.
        """
        return sorted(sum([self.channel_map[key]['tetrode_nrs'] for key in self.channel_map], []))

    @property
    def first_original_timestamp(self):
        return self._first_original_timestamp

    @property
    def last_original_timestamp(self):
        return self._last_original_timestamp

    @property
    def microvolt_gain(self):
        """Multiplier for int16 signal values (waveforms and continuous)
        to convert them to microvolts.

        :return: gain
        :rtype: float
        """
        return self._microvolt_gain

    @property
    def waveform_sampling_rate(self):
        """Sampling rate of the first available unit.
        Assumes all units have same sampling rate.
        """
        if self._waveform_sampling_rate is None and not (self.units is None):
            for unit in self.units:
                if not (unit is None) and 'sampling_rate' in unit:
                    self._waveform_sampling_rate = unit['sampling_rate']
                    break
        return self._waveform_sampling_rate

    @property
    def continuous(self):
        return self._continuous

    @property
    def units(self):
        return self._units

    @property
    def unit_lookup_table(self):
        return copy(self._unit_lookup_table)

    @property
    def position(self):
        return self._position

    @property
    def position_data_sampling_rate(self):
        return copy(self._position_data_sampling_rate)

    @property
    def task_log_parser(self):
        return self._task_log_parser

    @property
    def task_data(self):
        return self._task_data

    @property
    def task_settings(self):
        return self._task_settings

    @property
    def analysis(self):
        return self._analysis

    @staticmethod
    def load_file_info(fpath):
        info = NWBio.extract_recording_info(fpath)
        return {'task': info['TaskName'] if info['TaskActive'] else None,
                'animal': info['animal'],
                'arena_size': info['arena_size'],
                'bad_channels': NWBio.listBadChannels(fpath),
                'channel_map': info['channel_map'],
                'experiment_id': info['experiment_id'],
                'continuous_data_duration': NWBio.get_recording_full_duration(fpath),
                'rec_datetime': convert_recording_time_string_to_datetime(info['time']),
                'file_datetime': datetime.fromtimestamp(os.path.getmtime(fpath))}

    @staticmethod
    def tetrode_channel_group(tetrode_nr, channel_map):
        for group in channel_map:
            if tetrode_nr in channel_map[group]['tetrode_nrs']:
                return group

    @staticmethod
    def __empty_analysis_dictionary():
        return {'epochs': {}}

    def load_continuous_downsampled_tetrode_data(self):
        """Loads downsampled continuous data from NWB file into self.continuous

        :return: None
        """
        if self.verbose:
            print('Loading continuous data for {}'.format(self.fpath))

        # Load downsampled data for all tetrodes in all channel groups
        tetrode_nrs = sum([self.channel_map[channel_group]['tetrode_nrs'] for channel_group in self.channel_map], [])
        tetrode_nrs.sort()
        self._continuous = NWBio.load_downsampled_tetrode_data_as_array(self.fpath, tetrode_nrs)

        # Crop data outside specified first and last timestamp
        idx = np.logical_and(self.continuous['timestamps'] > self.first_original_timestamp,
                             self.continuous['timestamps'] < self.last_original_timestamp)
        self.continuous['timestamps'] = self.continuous['timestamps'][idx, ...]
        self.continuous['continuous'] = self.continuous['continuous'][idx, ...]

        # Subtract first_original_timestamp offset from continuous data timestamps
        self.continuous['timestamps'] -= self.first_original_timestamp

        # Append channel_map channel_group name
        self.continuous['channel_group'] = []
        for tetrode_nr in self.continuous['tetrode_nrs']:
            channel_group = Recording.tetrode_channel_group(tetrode_nr, self.channel_map)
            if channel_group is None:
                raise Exception('tetrode_nr {} not found in channel_map'.format(tetrode_nr))
            self.continuous['channel_group'].append(channel_group)

        # Create 'analysis' field for continuous data
        self.continuous['analysis'] = {}

    def load_continuous_data(self, continuous_data_type):
        if continuous_data_type is None:
            return
        elif continuous_data_type == 'downsampled':
            self.load_continuous_downsampled_tetrode_data()
        elif continuous_data_type == 'raw':
            raise NotImplementedError

    def get_tetrode_cluster_id_unit(self, tetrode_nr, tetrode_cluster_id):
        """Returns spike_train corresponding to tetrode_nr and tetrode_cluster_id

        :param int tetrode_nr: tetrode_nr where spike_train was detected
        :param int tetrode_cluster_id: cluster_id of the spike_train on this tetrode
        :return: spike_train
        :rtype: dict
        """
        if tetrode_nr in self.unit_lookup_table:
            if tetrode_cluster_id in self.unit_lookup_table[tetrode_nr]:
                return self.units[self.unit_lookup_table[tetrode_nr][tetrode_cluster_id]]

    def get_all_units_on_tetrode(self, tetrode_nr):
        """Returns a list of units that match the tetrode number

        :param int tetrode_nr:
        :return: list of units
        :rtype: list
        """
        return [unit for unit in self.units if unit['tetrode_nr'] == tetrode_nr]

    def load_units(self, spike_name, clustering_name,
                   no_waveforms=False, ignore_cluster_zero=False):
        """Loads spike data and organise into a list of units across all tetrodes

        :param str spike_name: spike field to load from NWB file. Default is 'spikes'. See NWBio.load_spikes docs.
        :param clustering_name: allows specifying curated clusterID set to use, see NWBio.load_spikes docs.
                                if None, non-curated clusterIDs will be loaded if available.
        :type clustering_name: str or None
        :param bool no_waveforms: if True, no waveforms are loaded. Default is False.
        :param bool ignore_cluster_zero: if True, clusterIDs==0 are ignored. Default is False.
        """

        data = NWBio.load_spikes(self.fpath, spike_name=spike_name, use_idx_keep=True, use_badChan=True,
                                 no_waveforms=no_waveforms, clustering_name=clustering_name,
                                 verbose=self.verbose)

        if self.verbose:
            print('Splitting tetrode spike data into units for {}'.format(self.fpath))

        for tet_data in (tqdm(data) if self.verbose else data):

            # Find spikes on tetrode that occur within specified first and last timestamp
            tetrode_spikes_within_limits_idx = np.logical_and(
                tet_data['timestamps'] > self.first_original_timestamp,
                tet_data['timestamps'] < self.last_original_timestamp
            )

            # If no clusterID associated with tetrode spikes, assign all spikes to cluster 0
            if not ('clusterIDs' in tet_data):
                tet_data['clusterIDs'] = np.zeros(tet_data['timestamps'].shape)

            # Get unique units on this tetrode
            unique_units = np.unique(tet_data['clusterIDs'])

            # Create SpikeTrains and Unit classes for each cluster
            for tetrode_cluster_id in unique_units:

                if ignore_cluster_zero and tetrode_cluster_id == 0:
                    break

                # Filter spikes for this tetrode_cluster_id
                spike_idx = tet_data['clusterIDs'] == tetrode_cluster_id

                # Filter spikes by timestamp being within limits
                spike_idx = np.logical_and(spike_idx, tetrode_spikes_within_limits_idx)

                # Append spikes from this cluster to self.units list
                self.units.append(
                    {
                        'timestamps': tet_data['timestamps'][spike_idx] - self.first_original_timestamp,
                        'waveforms': None if no_waveforms else tet_data['waveforms'][spike_idx, ...],
                        'start_timestamp': 0.0,
                        'end_timestamp': self.info['last_timestamp'],
                        'sampling_rate': NWBio.OpenEphys_SamplingRate(),
                        'tetrode_nr': tet_data['nr_tetrode'],
                        'tetrode_cluster_id': tetrode_cluster_id,
                        'channel_group': Recording.tetrode_channel_group(
                            tet_data['nr_tetrode'], self.channel_map),
                        'analysis': {}  # Empty dict to append analysis results to each unit
                    }
                )

                # Include tetrode_cluster_id in self.unit_lookup_table for access of spike_trains
                # via tetrode_nr and tetrode_cluster_id
                if not (tet_data['nr_tetrode'] in self._unit_lookup_table):
                    self._unit_lookup_table[tet_data['nr_tetrode']] = {}
                self._unit_lookup_table[tet_data['nr_tetrode']][tetrode_cluster_id] = len(self.units) - 1

    @staticmethod
    def fill_data_gaps_with_nans(timestamps, data, sampling_rate, max_jitter=0.00001):
        """Fills gaps in data with np.nan values.

        Assumes that there are no samples at higher rate than sampling_rate.

        :param timestamps: timestamps of each datapoint value in first dimension
        :param np.array data: datapoints corresponding to timestamps along first dimension
                              dtype must be either np.float16, np.float32 or similar, to supprot np.nan
        :param int sampling_rate: sampling rate of data when missing values ignored
        :param float max_jitter: maximum seconds to allow error from sampling_rate before gaps being filled
        :return: timestamps, data
        :rtype: same as input
        """

        sampling_period = 1. / float(sampling_rate)

        # Find indices with gaps and gap sizes
        gaps = list(np.where(np.diff(timestamps) > (sampling_period + max_jitter))[0])
        gap_lengths = [float(timestamps[i + 1] - timestamps[i]) for i in gaps]
        gap_samples = [(int(round(l / sampling_period)) - 1) for l in gap_lengths]

        # Fill gaps with np.nan elements starting from the last
        for gi, gl in zip(gaps[::-1], gap_samples[::-1]):
            timestamps = np.insert(timestamps, gi + 1,
                                   np.linspace(timestamps[gi], timestamps[gi + 1], gl + 2)[1:-1])
            nans = np.full(gl, np.nan) if len(data.shape) == 1 else np.full((gl,) + data.shape[1:], np.nan)
            data = np.insert(data, gi + 1, nans, axis=0)

        return timestamps, data

    @staticmethod
    def drop_samples_in_position_data_preceding_negative_time_jumps(data):
        """This function recursively removes all forward jumps that result in reverse jumps
        in time according to timestamps.

        :param numpy.ndarray data: output from :py:func:`NWBio.load_processed_tracking_data`
        :return: same as `data` but with negative time negative time jumps removed
        """
        negative_jumps = np.where(np.diff(data[:, 0]) < 0)[0]
        if len(negative_jumps) == 0:
            return data
        else:
            data = np.delete(data, np.where(np.diff(data[:, 0]) < 0)[0], axis=0)
            return Recording.drop_samples_in_position_data_preceding_negative_time_jumps(data)

    @staticmethod
    def compute_movement_direction(xy):
        vectors = np.concatenate((np.array([[0, 0]], dtype=xy.dtype), xy[1:, :] - xy[:-1, :]), axis=0)
        complex_representation = np.array(list(map(lambda x: complex(*x), list(vectors))))
        return np.angle(complex_representation)

    def load_position_data(self, position_data_name='ProcessedPos', sampling_rate=30):
        """Loads position data from the recording and ensures constant sampling rate.
        Missing values are set to np.nan

        :param position_data_name: Name of position data to load. See NWBio.load_processed_tracking_data docs.
                                   If None, no position data is loaded.
        :type position_data_name: str or None
        :param int sampling_rate: sampling rate of position data in Hz
        """
        if position_data_name is None:
            return

        if self.verbose:
            print('Loading position data for {}'.format(self.fpath))

        data = NWBio.load_processed_tracking_data(self.fpath, subset=position_data_name)
        data = Recording.drop_samples_in_position_data_preceding_negative_time_jumps(data)
        timestamps = data[:, 0]
        xy = data[:, 1:3]

        # Check if sampling rate is likely to be correct
        if ((timestamps[-1] - timestamps[0]) / float(len(timestamps)) > (1. / float(sampling_rate)) * 1.01
                or (timestamps[-1] - timestamps[0]) / float(len(timestamps)) < (1. / float(sampling_rate)) * 0.99):
            warnings.warn('''Sampling rate in position data {} is unlikely to match {}
                             This check can raise a warning also if tracking data has large gaps in samples'''.
                          format(position_data_name, sampling_rate))

        # Fill any gaps in sampling with np.nan values
        timestamps, xy = Recording.fill_data_gaps_with_nans(timestamps, xy, sampling_rate)

        # Compute instantaneous speed
        speed = np.sqrt(np.diff(xy[:, 0])**2 + np.diff(xy[:, 1])**2) * sampling_rate
        speed = np.concatenate((np.array([speed[0]], dtype=speed.dtype), speed))

        self._position = {
            'timestamps': timestamps - self.first_original_timestamp,
            'xy': xy,
            'speed': speed,
            'movement_direction': self.compute_movement_direction(xy),
            'position_data_name': position_data_name,
            'sampling_rate': sampling_rate,
            'analysis': {}  # Empty dict to append analysis results to position data
        }

    def get_continuous_signal_for_channel_group(self, channel_group):
        """Returns continuous signal for channels that belong to specific channel_group

        :param channel_group:
        :return:
        """
        ind = indices_where_tetrode_nrs_in_tetrode_nrs(self.channel_map[channel_group]['tetrode_nrs'],
                                                       self.continuous['tetrode_nrs'])
        return self.continuous['continuous'][:, ind]

    def get_units_for_one_tetrode(self, tetrode_nr):
        return [unit for unit in self.units if unit['tetrode_nr'] == tetrode_nr]

    def get_task_data_first_timestamps(self, major_key, minor_key):
        """Returns the start timestamps of each epoch in self.task_data[major_key][minor_key]

        :param str major_key: primary task data key
        :param str minor_key: secondary task data key
        :return: timestamps of when the epochs started
        :rtype: list
        """
        return [x[0] for x in self.task_data[major_key][minor_key]['timestamps']]

    def get_task_data_epoch_duration(self, major_key, minor_key):
        """Returns the durations of each epoch in self.task_data[major_key][minor_key]

        :param str major_key: primary task data key
        :param str minor_key: secondary task data key
        :return: durations of each epoch
        :rtype: list
        """
        return [x[1] - x[0] for x in self.task_data[major_key][minor_key]['timestamps']]

    def get_task_data_rewards_across_feeders(self, reward_name):
        """Returns timestamps, quantities and feeder_ids for specified rewards across all feeders

        :param str reward_name: self.task_data['Reward'][reward_name]
        :return: {'timestamps', 'quantities', 'feeder_ids'}
        :rtype: dict
        """
        # Combine reward events across feeders
        timestamps = []
        amount = []
        feeder_ids = []
        for feeder_id in self.task_data['Reward'][reward_name]:
            timestamps += self.task_data['Reward'][reward_name][feeder_id]['timestamps']
            amount += map(float, self.task_data['Reward'][reward_name][feeder_id]['data'])
            feeder_ids += [feeder_id] * len(self.task_data['Reward'][reward_name][feeder_id]['timestamps'])

        # Sort combined events by timestamp
        timestamps, amount, feeder_ids = zip(*[(x, y, z) for x, y, z in sorted(zip(timestamps, amount, feeder_ids))])

        return {'timestamps': timestamps, 'quantities': amount, 'feeder_ids': feeder_ids}

    @staticmethod
    def correct_timestamps_in_log_parser_data(data, first_original_timestamp, key=''):
        """Subtracts first_original_timestamp from all timestamps in data

        Recursively works through each level in dictionary and if encounters
        'timestamps' element, subtracts first_original_timestamp from each value.

        :param data: task specific LogParser.data
        :param float first_original_timestamp: value to be subtracted from each encountered timestamp
        :param str key: used in recursive calls to pass on previous dictionary key
        :return: data with timestamps corrected
        :rtype: dict
        """
        if key == 'timestamps' and isinstance(data, list):
            for i, element in enumerate(data):
                if isinstance(element, list):
                    data[i] = Recording.correct_timestamps_in_log_parser_data(element, first_original_timestamp,
                                                                              key=key)
                else:
                    data[i] = element - first_original_timestamp
        elif isinstance(data, dict):
            for key in data:
                data[key] = Recording.correct_timestamps_in_log_parser_data(data[key], first_original_timestamp,
                                                                            key=key)

        return data

    @staticmethod
    def fix_incomplete_final_timestamps_in_log_parser_data(data, recording_last_timepoint, key=''):
        """Fixes final timestamps missing in task data epochs.

        Recursively works through each level in dictionary and if encounters
        'timestamps' element with two values in the first timestamp element (i.e. it's an epoch),
        then appends the recording_last_timepoint as the second value in the final element if missing.

        :param data: task specific LogParser.data
        :param float recording_last_timepoint: value to be added as final timestamp of last epoch where missing
        :param str key: used in recursive calls to pass on previous dictionary key
        :return: data with timestamps corrected
        :rtype: dict
        """
        if key == 'timestamps' and isinstance(data, list):
            if isinstance(data[0], list) and len(data[0]) == 2:
                if len(data[-1]) == 1:
                    data[-1].append(recording_last_timepoint)
        elif isinstance(data, dict):
            for key in data:
                data[key] = Recording.fix_incomplete_final_timestamps_in_log_parser_data(
                    data[key], recording_last_timepoint, key=key
                )

        return data

    def load_task_data(self):
        """Loads task data into self.task_data and retains LogParser at self.task_log_parser
        """

        if self.verbose:
            print('Loading task data for {}'.format(self.fpath))

        # Get LogParser class for this recording, initialized with recording event data
        self._task_log_parser = NWBio.get_recording_log_parser(self.fpath, final_timestamp=self.last_original_timestamp)

        if not (self._task_log_parser is None):
            self._task_data = Recording.correct_timestamps_in_log_parser_data(deepcopy(self.task_log_parser.data),
                                                                              self.first_original_timestamp)
            self._task_data = Recording.fix_incomplete_final_timestamps_in_log_parser_data(
                self._task_data, self.info['last_timestamp']
            )

    def load_task_settings(self):
        """Loads task settings into self.task_settings
        """
        self._task_settings = NWBio.load_settings(self.fpath, '/TaskSettings/')

    def get_smoothed_speed(self, size, method='gaussian'):
        """Returns smoothed speed.

        :param float size: smoothing kernel size in seconds. Rounded to nearest integer to specify number of bins.
            Defined the sigma, if `method='gaussian'`; length of kernel, if `method='boxcar'`.
        :param str method: 'gaussian' for :py:func:`scipy.ndimage.gaussian_filter1d` (default)
                           'boxcar' for :py:func:`scipy.ndimage.uniform_filter1d`
        :param int order: number of times to apply smoothing (see `method` description specified above.
        :return: speed_smoothed shape (n_samples,)
        :rtype: numpy.ndarray
        """
        if self.position is None or not ('speed' in self.position):
            raise Exception('position data or speed is not available for {}'.format(self.fpath))

        if method == 'boxcar':
            filter_func = uniform_filter1d
        elif method == 'gaussian':
            filter_func = gaussian_filter1d
        else:
            raise ValueError('Unknown smoothing method {}'.format(method))

        n_smooth_bins = int(np.round(size * self.position['sampling_rate']))

        return filter_func(self.position['speed'], n_smooth_bins)

    def get_position_indices_matching_timestamps(self, timestamps):
        """Returns indices to `Recording.position['xy']`, `Recording.position['timestamps']`
        and `Recording.position['speed']` that are closest to provided timestamps.

        :param numpy.ndarray timestamps: shape (N,)
        :return: indices
        :rtype: numpy.ndarray
        """
        return closest_argmin(timestamps, self.position['timestamps'])

    def edit_info(self):
        """Returns self._info that can be edited, unlike :py:attr:`Recording.info` that returns a copy.

        :return: self._info dictionary that can be edited
        :rtype: dict
        """
        return self._info

    def save_analysis(self, overwrite_mode='partial', verbose=False):
        """Stores data in analysis dictionaries into NWB file.

        See :py:func:`NWBio.save_analysis` and :py:func:`NWBio.recursively_save_dict_contents_to_group`
        for details on supported data structures.

        Recording.analysis
        Recording.units[n]['analysis'] for n in range(len(units))
        Recording.continuous['analysis']
        Recording.position['analysis']

        :param str overwrite_mode: 'partial', 'full' or 'none'. Default is 'partial'.
            See `openEPhys_DACQ.NWBio.save_analysis` parameters `overwrite` and
            `complete_overwrite` for meaning of 'partial' and 'full', respectively.
        """

        # Ensure 'units', 'continuous' and 'position' fields do not exist in self.analysis
        if 'units' in self.analysis:
            raise Exception('units field must not be used in analysis dictionary')
        if 'continuous' in self.analysis:
            raise Exception('continuous field must not be used in analysis dictionary')
        if 'position' in self.analysis:
            raise Exception('position field must not be used in analysis dictionary')

        # Combine unit specific analysis to the dictionary
        self.analysis['units'] = []
        for unit in self.units:
            self.analysis['units'].append(
                {'record_validation': unit_analysis_record_validation_dict(unit),
                 'analysis': unit['analysis']}
            )
        # Combine continuous data specific analysis to the dictionary
        if not (self.continuous is None):
            self.analysis['continuous'] = {
                'record_validation': continuous_record_validation_dict(self.continuous),
                'analysis': self.continuous['analysis']
            }
        # Combine position data specific analysis to the dictionary
        if not (self.position is None):
            self.analysis['position'] = {
                'record_validation': position_record_validation_dict(self.position),
                'analysis': self.position['analysis']
            }

        # Save analysis data into NWB file
        if overwrite_mode == 'partial':
            NWBio.save_analysis(self.fpath, self.analysis, overwrite=True, verbose=verbose)
        elif overwrite_mode == 'full':
            NWBio.save_analysis(self.fpath, self.analysis, complete_overwrite=True, verbose=verbose)
        elif overwrite_mode == 'none':
            NWBio.save_analysis(self.fpath, self.analysis, overwrite=False, complete_overwrite=False,
                                verbose=verbose)
        else:
            raise ValueError('Unknown overwrite_mode {}'.format(overwrite_mode))

        # Remove 'units', 'continuous' and 'position' fields from self.analysis
        if 'units' in self.analysis:
            del self.analysis['units']
        if 'continuous' in self.analysis:
            del self.analysis['continuous']
        if 'position' in self.analysis:
            del self.analysis['position']

    def load_analysis(self, ignore=()):
        """Loads analysis data from NWB file to this Recording instance.

        See :py:func:`Recording.save_analysis` for fields that will be loaded and overwritten.

        .. warning:: any existing analysis data in those fields in this Recording instance will be lost.

        :param tuple ignore: any paths through the loaded dictionaries that contain a key matching
            any element in ignore will not be loaded and the dictionary tree will terminate there with None.
        """

        if self.verbose:
            print('Loading analysis for {}'.format(self.fpath))

        analysis = NWBio.load_analysis(self.fpath, ignore=ignore)

        if self.verbose:
            print('Ensuring that loaded data matches with current Recording instance.')

        # Check that loaded units match with those in this Recording instance
        if len(self.units) > 0 and 'units' in analysis:
            if len(analysis['units']) != len(self.units):
                raise ValueError('loaded units analysis data does not fit with units in this Recording instance.')
            for l_unit, r_unit in zip(analysis['units'], self.units):
                if not check_if_dictionaries_match(l_unit['record_validation'],
                                                   unit_analysis_record_validation_dict(r_unit)):
                    raise ValueError('loaded units analysis data does not fit with units \n'
                                     + 'in this Recording instance.')
                del l_unit['record_validation']

        # Check that loaded continuous data analysis matches with data in this Recording instance
        if not (self.continuous is None) and 'continuous' in analysis:
            if not check_if_dictionaries_match(analysis['continuous']['record_validation'],
                                               continuous_record_validation_dict(self.continuous)):
                raise ValueError('loaded continuous analysis data does not fit with continuous data \n'
                                 + 'in this Recording instance.')
            del analysis['continuous']['record_validation']

        # Check that loaded position data analysis matches with data in this Recording instance
        if not (self.position is None) and 'position' in analysis:
            if not check_if_dictionaries_match(analysis['position']['record_validation'],
                                               position_record_validation_dict(self.position)):
                raise ValueError('loaded position analysis data does not fit with position data \n'
                                 + 'in this Recording instance.')
            del analysis['position']['record_validation']

        # Move units, continuous and position data analysis to correct dictionaries
        if len(self.units) > 0 and 'units' in analysis:
            for l_unit, r_unit in zip(analysis['units'], self.units):
                r_unit['analysis'] = l_unit['analysis'] if 'analysis' in l_unit else {}
        if not (self.continuous is None) and 'continuous' in analysis:
            self.continuous['analysis'] = \
                analysis['continuous']['analysis'] if 'analysis' in analysis['continuous'] else {}
        if not (self.position is None) and 'position' in analysis:
            self.position['analysis'] = \
                analysis['position']['analysis'] if 'analysis' in analysis['position'] else {}

        if self.verbose:
            print('Inserting loaded analysis data into Recording instance.')

        # Remove 'units', 'continuous' and 'position' fields from analysis
        if 'units' in analysis:
            del analysis['units']
        if 'continuous' in analysis:
            del analysis['continuous']
        if 'position' in analysis:
            del analysis['position']

        # Add loaded analysis items to self.analysis
        self._analysis = Recording.__empty_analysis_dictionary()
        if analysis:
            for key, item in analysis.items():
                self._analysis[key] = item

        del analysis


class Recordings(object):
    """
    Loads and combines multiple :class:`Recording` instances.

    Allows accessing same units and recording channels across multiple class:`Recording` instances.

    class:`Recordings` instance has same major properties as class:`Recording`, but these are lists
    referring to matching properties of each class:`Recording` instance, particularly :py:attr:`~units`

    If a class:`Recording` instance does not have a particular unit, it is None in :py:attr:`~units`.

    :py:class:`Recordings` also acts as an iterator, for example the following code yields
    iteratively all individual :py:class:`Recording` instances that are part of the :py:class:`Recordings`
    instance:

    >>> recordings = Recordings(['/data1/filename', '/data2/filename'])
    >>> for recording in recordings:
    >>>     print(recording.fpath)
    /data1/filename
    /data2/filename

    The individual :py:class:`Recording` instances can also be accessed at :py:attr:`Recordings.recordings`
    or through indexing into :py:class:`Recordings`, such as `recordings[2]` or recordings[-1]`.

    Assumes all Recording classes loaded have consistent unit identities
    and same channel mapping.

    See class:`Recording` documentation for further specification data format.
    """

    def __init__(self, fpaths, name=None, *args, **kwargs):
        """Loads each recording in :class:`Recording` instance and ensures data is compatible.

        :param list fpaths: list of strings as fpath argument for each each :class:`Recording` instance
        :param args: Same arguments are used for each :class:`Recording` instance.
        :param kwargs: Same arguments are used for each :class:`Recording` instance.
        """

        if 'verbose' in kwargs and kwargs['verbose']:
            self.verbose = True
        else:
            self.verbose = False

        self._fpaths = [parse_fpath(fpath) for fpath in fpaths]

        self._name = name

        if self.verbose:
            print('Loading all recordings {}'.format(self.fpaths))

        self._recordings = [Recording(fpath, *args, **kwargs) for fpath in self.fpaths]

        # Verify that recordings are compatible
        if self.verbose:
            print('Verifying recording compatibility')
        self._verify_all_recordings_have_compatible_continuous_data()
        self._verify_all_recordings_have_compatible_position_data()
        self._verify_all_recordings_have_compatible_channel_map()

        # Create a list of units across tetrodes
        if self.verbose:
            print('Combining units across recordings')
        self._units = []
        self._unit_tetrode_nrs = []
        self._unit_tetrode_cluster_ids = []
        for tetrode_nr, tetrode_cluster_ids in zip(*self._get_list_of_cluster_ids_for_tetrodes_across_recordings()):
            for tetrode_cluster_id in tetrode_cluster_ids:
                self.units.append(
                    [recording.get_tetrode_cluster_id_unit(tetrode_nr, tetrode_cluster_id)
                     for recording in self.recordings]
                )
                self._unit_tetrode_nrs.append(tetrode_nr)
                self._unit_tetrode_cluster_ids.append(tetrode_cluster_id)

        # Create unknown variables
        self._waveform_sampling_rate = None

    def __iter__(self):
        return iter(self.recordings)

    def __getitem__(self, key):
        return self.recordings[key]

    def __len__(self):
        return len(self.recordings)

    @property
    def fpaths(self):
        return self._fpaths

    @property
    def name(self):
        return copy(self._name)

    @property
    def recordings(self):
        return self._recordings

    @property
    def fpath(self):
        return [recording.fpath for recording in self.recordings]

    @property
    def info(self):
        return [recording.info for recording in self.recordings]

    @property
    def channel_map(self):
        # Use channel_map attribute from first recording as they are all same
        return self.recordings[0].channel_map

    @property
    def continuous(self):
        return [recording.continuous for recording in self.recordings]

    @property
    def units(self):
        return self._units

    @property
    def unit_tetrode_nrs(self):
        return self._unit_tetrode_nrs

    @property
    def unit_tetrode_cluster_ids(self):
        return self._unit_tetrode_cluster_ids

    @property
    def position(self):
        return [recording.position for recording in self.recordings]

    @property
    def task_data(self):
        return [recording.task_data for recording in self.recordings]

    @property
    def waveform_sampling_rate(self):
        """waveform_sampling_rate attribute of the first Recording instance.
        """
        return self.recordings[0].waveform_sampling_rate

    @property
    def microvolt_gain(self):
        """microvolt_gain attribute of the first Recording instance.

        :return: gain
        :rtype: float
        """
        return self.recordings[0].microvolt_gain

    def _verify_all_recordings_have_compatible_channel_map(self):
        # Ensure that channel_map matches across segments
        for recording in self.recordings[1:]:
            if not NWBio.check_if_channel_maps_are_same(recording.channel_map, self.recordings[0].channel_map):
                raise ValueError('The channel_map does not match in all recordings')

    def _verify_all_recordings_have_compatible_continuous_data(self):
        for recording in self.recordings[1:]:
            if not isinstance(self.recordings[0].continuous, type(recording.continuous)):
                raise Exception('tetrode_nrs of continuous data does not match in all recordings.')
            if not (self.recordings[0].continuous is None):
                if self.recordings[0].continuous['tetrode_nrs'] != recording.continuous['tetrode_nrs']:
                    raise Exception('tetrode_nrs of continuous data does not match in all recordings.')
                if self.recordings[0].continuous['channels'] != recording.continuous['channels']:
                    raise Exception('channels of continuous data does not match in all recordings.')
                if self.recordings[0].continuous['channel_group'] != recording.continuous['channel_group']:
                    raise Exception('channel_group of continuous data does not match in all recordings.')

    def _verify_all_recordings_have_compatible_position_data(self):
        for recording in self.recordings[1:]:
            if not isinstance(self.recordings[0].position, type(recording.position)):
                raise Exception('position_data_name does not match in all recordings.')
            if not (self.recordings[0].position is None):
                if self.recordings[0].position['position_data_name'] != recording.position['position_data_name']:
                    raise Exception('position_data_name does not match in all recordings.')
                if self.recordings[0].position_data_sampling_rate != recording.position_data_sampling_rate:
                    raise Exception('position_data_sampling_rate does not match in all recordings.')

    def _get_list_of_cluster_ids_for_tetrodes_across_recordings(self):
        """Finds unique cluster_ids for all tetrodes across all recordings

        :return: tetrode_nrs, cluster_ids
                 tetrode_nrs lists all tetrodes for which spiketrains were present in dataset.
                 cluster_ids is a list where each element corresponds to tetrode_nrs and lists
                 unique cluster_ids that occurred on that tetrode across all recordings.
        :rtype: list, list
        """

        tetrode_nrs = []
        cluster_ids = []
        for recording in self.recordings:
            for unit in recording.units:
                if not unit['tetrode_nr'] in tetrode_nrs:
                    tetrode_nrs.append(unit['tetrode_nr'])
                    cluster_ids.append([])
                tetrode_nr_idx = tetrode_nrs.index(unit['tetrode_nr'])
                if not unit['tetrode_cluster_id'] in cluster_ids[tetrode_nr_idx]:
                    cluster_ids[tetrode_nr_idx].append(unit['tetrode_cluster_id'])

        return tetrode_nrs, cluster_ids

    def get_list_of_tetrode_nrs_across_recordings(self):
        """Returns a list of tetrode_nrs that occur in any of the :py:class:`Recording`
        instances associated with this class.

        :return: tetrode_nrs
        :rtype: list
        """
        return sorted(list(set(self.unit_tetrode_nrs)))

    def get_units_for_one_tetrode(self, tetrode_nr):
        """Returns units from :py:attr:`Recordings.units` that are associated with tetrode_nr

        :param int tetrode_nr:
        :return: list of units
        :rtype: list
        """
        return [unit for unit, unit_tetrode_nrs in zip(self.units, self.unit_tetrode_nrs)
                if unit_tetrode_nrs == tetrode_nr]

    def get_units_for_one_tetrode_as_one_level_list(self, tetrode_nr):
        """Returns units from :py:attr:`Recordings.units` that are associated with tetrode_nr
        in a one level list and the unit group identity for each element in resulting list.
        The unit group identity should in general case group same units across recording instances.

        This is a convenience method for cases such as processing the waveforms of units
        on the same tetrode concurrently across mutliple recordings.
        For example, with :py:class:`spikes.waveforms.RecordingUnitListWaveformProperties`

        :param int tetrode_nr:
        :return: unit_list, unit_group_ind
        :rtype: list, list
        """
        return (sum(self.get_units_for_one_tetrode(tetrode_nr), []),
                sum([[i] * len(unit) for i, unit in enumerate(self.units)], []))

    def save_analysis(self, overwrite_mode='partial', verbose=False):
        """Calls :py:func:`Recording.save_analysis` for each Recording class.

        :param str overwrite_mode: 'partial', 'full' or 'none'. Default is 'partial'.
            See `openEPhys_DACQ.NWBio.save_analysis` parameters `overwrite` and
            `complete_overwrite` for meaning of 'partial' and 'full', respectively.
        """
        for recording in self.recordings:
            recording.save_analysis(overwrite_mode=overwrite_mode, verbose=verbose)

    def load_analysis(self, *args, **kwargs):
        """Calls :py:func:`Recording.load_analysis` for each Recording class.

        All input arguments are passed on to each call of :py:func:`Recording.load_analysis`.
        """
        for recording in self.recordings:
            recording.load_analysis(*args, **kwargs)

    def first_available_recording_unit(self, i):
        """Returns the unit dictionary of this unit in first recording where it was detected.

        :param int i: unit index
        :return: unit_dict
        :rtype: dict
        """
        for unit in self.units[i]:
            if not (unit is None):
                return unit

    def check_if_all_recordings_have_same_position_sampling_rate(self):
        return all([self.position[0]['sampling_rate'] == x['sampling_rate'] for x in self.position[1:]])

    def unit_timestamps_concatenated_across_recordings(self, i, position_sample_gap=True):
        """Returns timestamps of a unit concatenated across recordings, with temporal shifts
        to spikes times of sequential recordings based on duration of previous recordings.

        :param int i: unit position in list :py:attr:`Recordings.units`
        :param bool position_sample_gap: if True (default), spike timestamps of consecutive recordings
            are also shifted by one position sampling period. This simplifies concatenating spike and position
            data across recordings while maintaining their alignment.
        :return: concatenated_timestamps
        :rtype: numpy.ndarray
        """

        if position_sample_gap and not self.check_if_all_recordings_have_same_position_sampling_rate():
            raise Exception('All recordings must have same position sampling rate to use position_sample_gap\n'
                            + 'in method unit_timestamps_concatenated_across_recordings')
        position_sample_shift = 1. / float(self.position[0]['sampling_rate']) if position_sample_gap else 0.

        spiketimes = []

        for i_recording, unit in enumerate(self.units[i]):

            if not (unit is None):
                spiketimes.append(unit['timestamps']
                                  + position_sample_shift * i_recording
                                  + sum([recording.info['last_timestamp']
                                         for recording in self.recordings[:i_recording]]))

        return np.concatenate(spiketimes, axis=0)

    def position_data_concatenated_across_recordings(self):
        """Returns position data concatenated across recordings in the same format as recordings.position[0],
        but with empty analysis dictionary.
        Timestamps of position data from sequential recordings are shifted based on duration of previous
        recordings and by position sample period between each recording.

        :return: position_data
        :rtype: dict
        """

        if not self.check_if_all_recordings_have_same_position_sampling_rate():
            raise Exception('All recordings must have same position sampling rate to concatenate position data.')
        position_sample_shift = 1. / float(self.position[0]['sampling_rate'])

        xy = []
        speed = []
        timestamps = []

        for i_recording, recording_position in enumerate(self.position):

            xy.append(recording_position['xy'])
            speed.append(recording_position['speed'])

            timestamps.append(recording_position['timestamps']
                              + position_sample_shift * i_recording
                              + sum([recording.info['last_timestamp']
                                     for recording in self.recordings[:i_recording]]))

        return {
            'xy': np.concatenate(xy, axis=0),
            'speed': np.concatenate(speed),
            'timestamps': np.concatenate(timestamps),
            'position_data_name': self.position[0]['position_data_name'],
            'sampling_rate': self.position[0]['sampling_rate'],
            'analysis': {}
        }

    def get_position_index_in_concatenated_position_data(self, i_recording, position_index):
        """Returns the index in concatenated position data corresponding to a position data index in
        a specific recording.

        Assumes all recordings have same position sampling rate as concatenated position data is being used.

        :param i_recording:
        :param position_index:
        :return: position_index_in_concatenated_data
        :rtype: int
        """
        return sum([x['xy'].shape[0] for x in self.position[:i_recording]]) + position_index

    @staticmethod
    def count_spikes_in_unit_list(unit_list):
        """Returns total count of spikes in each element of unit_list.

        :param unit_list: list of unit dictionaries
        :return: count
        :rtype: tuple
        """
        if len(unit_list) == 0:
            raise ValueError('There must be at least one unit in the list.')
        n_spikes = []
        for unit in unit_list:
            if unit is None:
                n_spikes.append(0)
            else:
                n_spikes.append(unit['timestamps'].size)

        return tuple(n_spikes)

    def mean_firing_rate_of_unit(self, i_unit):
        """Returns the mean firing rate of specified unit across all recordings in this Recordings instance.

        :param int i_unit: index of unit in Recordings.units
        :return: mean_firing_rate
        :rtype: float
        """
        return (sum(self.count_spikes_in_unit_list(self.units[i_unit]))
                / float(sum([info['cropped_duration'] for info in self.info])))
