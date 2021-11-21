import argparse
import os
from copy import deepcopy
import numpy as np
from multiprocessing import Process
from tqdm import tqdm
import pandas as pd
from elephant import spectral


from barrylab_ephys_analysis.blea_utils import argparse_to_kwargs
from barrylab_ephys_analysis.recording_io import check_if_analysis_field_in_file, Recordings
from barrylab_ephys_analysis.spikes.waveforms import compute_all_waveform_properties_for_all_units_in_recordings
from barrylab_ephys_analysis.spikes.sorting import (compute_all_quality_measures_for_all_units_in_recordings,
                                                    UnitDuplicateDetector)
from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap

from barrylab_ephys_analysis.spatial.similarity import spatial_correlation
from barrylab_ephys_analysis.spikes.correlograms import (create_correlation_bin_edges,
                                                         auto_correlation)
from barrylab_ephys_analysis.spatial.fields import detect_fields, compute_field_stability
from barrylab_ephys_analysis.models.decoding import FlatPriorBayesPositionDecoding

from barrylab_ephys_analysis.lfp.oscillations import (FrequencyBandAmplitude,
                                                      FrequencyBandFrequency,
                                                      FrequencyBandPhase)

from barrylab_ephys_analysis.scripts.exp_scales import load
from barrylab_ephys_analysis.scripts.exp_scales.params import Params
from barrylab_ephys_analysis.scripts.exp_scales import snippets


def load_recordings_and_correct_experiment_ids(fpaths, verbose=False, **kwargs):
    recordings = Recordings(fpaths, clustering_name=Params.clustering_name,
                            verbose=verbose, **kwargs)
    snippets.rename_last_recording_a2(recordings)
    recordings.load_analysis()

    return recordings


def compute_waveform_properties_and_sorting_quality_if_not_available(
        recordings=None, fpaths=None, recompute=False, verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        waveform_properties_availability = [False]
        sorting_quality_availability = [False]
    else:
        # Check which data is available in file
        waveform_properties_availability = []
        sorting_quality_availability = []
        for fpath in fpaths:
            waveform_properties_availability.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/waveform_properties'
                )
            )
            sorting_quality_availability.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/sorting_quality'
                )
            )

    if not all(waveform_properties_availability) or not all(sorting_quality_availability):

        if verbose:
            print('Computing waveform properties and/or sorting quality')

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        if not all(waveform_properties_availability):
            compute_all_waveform_properties_for_all_units_in_recordings(recordings, verbose=verbose)

        compute_all_quality_measures_for_all_units_in_recordings(
            recordings, feature_names=Params.sorting_quality_feature_names, verbose=verbose
        )

    return recordings


def compute_ratemap_speed_mask_if_not_available(recordings=None, fpaths=None, recompute=False,
                                                verbose=False, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        speed_mask_available = [False]
    else:
        # Check if data is available in file
        speed_mask_available = []
        for fpath in fpaths:
            speed_mask_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'position/analysis/ratemap_speed_mask'
                )
            )

    if not all(speed_mask_available):

        if verbose:
            print('Computing speed thresholded xy_mask.')

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        for recording in recordings:
            smoothed_speed = recording.get_smoothed_speed(Params.xy_masking['speed_smoothing_window'],
                                                          method=Params.xy_masking['speed_smoothing_method'])
            xy_mask = smoothed_speed >= Params.xy_masking['speed_threshold']
            recording.position['analysis']['ratemap_speed_mask'] = xy_mask

    return recordings


def compute_ratemaps_if_not_available(recordings=None, fpaths=None, recompute=False,
                                      verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        ratemap_available = [False]
    else:
        # Check if data is available in file
        ratemap_available = []
        for fpath in fpaths:
            ratemap_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/spatial_ratemaps'
                )
            )

    if not all(ratemap_available):

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        if verbose:
            print('Computing ratemaps')

        for recording in recordings:
            spatial_window = (0, recording.info['arena_size'][0], 0, recording.info['arena_size'][1])
            direction_filter_kwargs = None
            SpatialRatemap.compute_all_ratemaps_for_all_units_in_recording(
                recording, spatial_window=spatial_window,
                xy_mask=recording.position['analysis']['ratemap_speed_mask'],
                verbose=verbose, direction_filter_kwargs=direction_filter_kwargs,
                **Params.spatial_ratemap
            )

    return recordings


def compute_ratemap_stability_if_not_available(recordings=None, fpaths=None, recompute=False,
                                               verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        ratemap_stability_available = [False]
    else:
        # Check if data is available in file
        ratemap_stability_available = []
        for fpath in fpaths:
            ratemap_stability_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/spatial_ratemaps/spike_rates_halves/stability'
                )
            )
            ratemap_stability_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/spatial_ratemaps/spike_rates_minutes/stability'
                )
            )

    if not all(ratemap_stability_available):

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        if verbose:
            print('Computing ratemap stability')

        for recording in recordings:
            for unit in recording.units:

                unit['analysis']['spatial_ratemaps']['spike_rates_halves']['stability'] = \
                    np.array(
                        spatial_correlation(
                            unit['analysis']['spatial_ratemaps']['spike_rates_halves']['first'],
                            unit['analysis']['spatial_ratemaps']['spike_rates_halves']['second'],
                            **Params.ratemap_stability_kwargs
                        )[0]
                    )

                unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['stability'] = \
                    np.array(
                        spatial_correlation(
                            unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'],
                            unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even'],
                            **Params.ratemap_stability_kwargs
                        )[0]
                    )

    return recordings


def detect_fields_if_not_available(recordings=None, fpaths=None, recompute=False,
                                   verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        fields_available = [False]
    else:
        # Check if data is available in file
        fields_available = [
            check_if_analysis_field_in_file(
                fpaths[0],
                'fields_NWBLIST'
            )
        ]

    if not all(fields_available):

        if verbose:
            print('Detecting fields for {}'.format(fpaths))

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        recordings[0].analysis['fields'] = []

        animal_field_ind = 0

        for i_recording_unit, recordings_unit in enumerate(tqdm(recordings.units) if verbose else recordings.units):

            for recording, unit in zip(recordings, recordings_unit):

                if unit is None:
                    continue

                # Extract fields from cell ratemap
                field_ratemaps = detect_fields(
                    unit['analysis']['spatial_ratemaps']['spike_rates_smoothed'],
                    (unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'],
                     unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even']),
                    **Params.field_detection
                )

                if len(field_ratemaps) == 0:
                    continue

                # Fit gaussians to fields
                data_x_values, data_y_values = SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(
                    unit['analysis']['spatial_ratemaps']['spike_rates_smoothed'].shape,
                    unit['analysis']['spatial_ratemaps']['spatial_window'])

                for i, field_ratemap in enumerate(field_ratemaps):

                    # Compute field peak locations
                    field_ratemap_no_nan = np.ma.array(field_ratemap, mask=np.isnan(field_ratemap))
                    a, b = np.unravel_index(np.argmax(field_ratemap_no_nan), field_ratemap_no_nan.shape)
                    peak_x, peak_y = (data_x_values[b], data_y_values[a])

                    # Compute field_ratemap stability
                    stability_halves = compute_field_stability(
                        field_ratemap,
                        unit['analysis']['spatial_ratemaps']['spike_rates_halves']['first'],
                        unit['analysis']['spatial_ratemaps']['spike_rates_halves']['second'],
                        Params.field_detection['secondary_filter_kwargs']['stability_kwargs']['min_included_value'],
                        Params.field_detection['secondary_filter_kwargs']['stability_kwargs']['min_bins']
                    )
                    stability_minutes = compute_field_stability(
                        field_ratemap,
                        unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'],
                        unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even'],
                        Params.field_detection['secondary_filter_kwargs']['stability_kwargs']['min_included_value'],
                        Params.field_detection['secondary_filter_kwargs']['stability_kwargs']['min_bins']
                    )

                    recordings[0].analysis['fields'].append({

                        'ratemap': field_ratemap,
                        'spatial_window': unit['analysis']['spatial_ratemaps']['spatial_window'],

                        'properties': {
                            'animal': recording.info['animal'],
                            'animal_field': animal_field_ind,
                            'animal_unit': i_recording_unit,
                            'experiment_id': recording.info['experiment_id'],
                            'stability_minutes': stability_minutes,
                            'stability_halves': stability_halves,
                            'peak_spike_rate': np.nanmax(field_ratemap.flatten()),
                            'median_spike_rate': np.nanmedian(field_ratemap.flatten()),
                            'peak_x': peak_x,
                            'peak_y': peak_y,
                            'area': np.float64(float(np.sum(~np.isnan(field_ratemap)))
                                               * (Params.spatial_ratemap['bin_size'] ** 2))
                        }
                    })

                    field_properties_dict = recordings[0].analysis['fields'][-1]['properties']

                    field_properties_dict['peak_nearest_corner'] = \
                        snippets.compute_distance_to_nearest_corner((field_properties_dict['peak_x'],
                                                                     field_properties_dict['peak_y']),
                                                                    recording.info['arena_size'])
                    field_properties_dict['peak_nearest_wall'] = \
                        snippets.compute_distance_to_nearest_wall((field_properties_dict['peak_x'],
                                                                   field_properties_dict['peak_y']),
                                                                  recording.info['arena_size'])

                    animal_field_ind += 1

    return recordings


def set_duplicate_category_to_noise(recordings, verbose=False):

    # Get list of duplicate units in this recording
    udd = UnitDuplicateDetector(recordings, verbose=verbose)
    duplicate_unit_pairs = udd.list_duplicates(**Params.duplicate_cell_criteria)

    # Iterate over all duplicate unit pairs
    for i, j in duplicate_unit_pairs:

        # Verify that both units are categorised as 'place_cell', otherwise skip rest of this loop iteration
        unit_i = [unit for unit in recordings.units[i] if not (unit is None)][0]
        unit_j = [unit for unit in recordings.units[j] if not (unit is None)][0]
        if unit_i['analysis']['category'] != 'place_cell' or unit_j['analysis']['category'] != 'place_cell':
            continue

        # Find unit with more spikes
        n_spikes_i = sum([len(unit['timestamps']) for unit in recordings.units[i] if not (unit is None)])
        n_spikes_j = sum([len(unit['timestamps']) for unit in recordings.units[j] if not (unit is None)])

        # Specify which unit to keep based on n_spikes. Keep the unit with fewer spikes (less likely contaminated).
        if n_spikes_i < n_spikes_j:
            good_unit_ind = i
            noise_unit_ind = j
        else:
            good_unit_ind = j
            noise_unit_ind = i

        # Set the category to 'noise' for the noise_unit_ind
        for unit in recordings.units[noise_unit_ind]:
            if not (unit is None):
                unit['analysis']['category'] = 'noise'

        if verbose:
            print('Unit {} set as noise. Unit {} kept as place cell'.format(noise_unit_ind, good_unit_ind))


def assign_unit_categories_if_not_available(recordings=None, fpaths=None, recompute=False,
                                            verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        ratemap_stability_available = [False]
    else:
        # Check if data is available in file
        ratemap_stability_available = []
        for fpath in fpaths:
            ratemap_stability_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/category'
                )
            )

    if not all(ratemap_stability_available):

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        if verbose:
            print('Assigning unit categories')

        create_df_fields_for_recordings(recordings)

        for i_recording_unit, recordings_unit in enumerate(recordings.units):

            unit = recordings_unit[snippets.get_index_where_most_spikes_in_unit_list(recordings_unit)]

            i_max_amp = np.argmax(unit['analysis']['mean_waveform_properties']['amplitude'])

            waveform_properties = unit['analysis']['mean_waveform_properties']

            max_stability_minutes = \
                np.nanmax([unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['stability']
                           for unit in recordings_unit if not (unit is None)])
            max_stability_halves = \
                np.nanmax([unit['analysis']['spatial_ratemaps']['spike_rates_halves']['stability']
                           for unit in recordings_unit if not (unit is None)])

            pc_criteria = Params.place_cell_criteria
            in_criteria = Params.interneuron_criteria

            if unit['tetrode_cluster_id'] == Params.noise_tetrode_cluster_id:

                category = 'noise'

            elif (
                    waveform_properties['peak_to_trough'][i_max_amp] >= pc_criteria['min_peak_to_trough']
                    and waveform_properties['half_width'][i_max_amp] >= pc_criteria['min_half_width']
                    and max_stability_minutes >= pc_criteria['min_stability_minutes']
                    and max_stability_halves >= pc_criteria['min_stability_halves']
                    and (sum(recordings.df_fields['animal_unit'] == i_recording_unit)
                         >= pc_criteria['min_number_of_fields'])
                    and recordings.mean_firing_rate_of_unit(i_recording_unit) <= pc_criteria['max_mean_firing_rate']
                    and waveform_properties['trough_ratio'][i_max_amp] >= pc_criteria['min_trough_ratio']
            ):

                category = 'place_cell'

            elif (
                    waveform_properties['half_width'][i_max_amp] < in_criteria['max_half_width']
                    and max_stability_halves < in_criteria['max_stability_halves']
                    and recordings.mean_firing_rate_of_unit(i_recording_unit) > in_criteria['min_mean_firing_rate']
                    and waveform_properties['trough_ratio'][i_max_amp] <= in_criteria['max_trough_ratio']
            ):

                category = 'interneuron'

            else:

                category = 'other_cell'

            for unit in recordings_unit:

                if not (unit is None):
                    unit['analysis']['category'] = category

        set_duplicate_category_to_noise(recordings, verbose=verbose)

    return recordings


def compute_unit_autocorrelations_if_not_available(recordings=None, fpaths=None, recompute=False,
                                                   verbose=True, timestamp_inter_recording_gap=60.,
                                                   **kwargs):
    """ Autocorrelation is computed by combining timestamps for this unit from all recordings.
    Timestamps of each recording are shifted forward by the duration of the previous recordings,
    plus timestamp_inter_recording_gap seconds to ensure no adjacent spikes are observed at recording edges.

    The same autocorrelogram information is stored in the unit dictionary of that unit in each recording.
    """

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        autocorrelations_available = [False]
    else:
        # Check if data is available in file
        autocorrelations_available = []
        for fpath in fpaths:
            autocorrelations_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'units_NWBLIST/0/analysis/autocorrelations'
                )
            )

    if not all(autocorrelations_available):

        if recordings is None:
            recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

        if verbose:
            print('Computing unit autocorrelations for animal {}'.format(recordings[0].info['animal']))

        for recordings_unit in tqdm(recordings.units):

            timestamps = []
            timestamps_shift = 0
            for i_recording, unit in enumerate(recordings_unit):

                if unit is None or unit['timestamps'].size < 2:
                    continue

                timestamps.append(unit['timestamps'] + timestamps_shift)

                timestamps_shift += timestamp_inter_recording_gap + recordings[i_recording].info['last_timestamp']

            if len(timestamps) == 0:
                continue

            timestamps = np.concatenate(timestamps)

            autocorrelations = {}

            for correlation_name in Params.autocorrelation_params:

                bin_edges = create_correlation_bin_edges(**Params.autocorrelation_params[correlation_name])

                autocorrelations[correlation_name] = {
                    'values': auto_correlation(timestamps, bin_edges, normalize=False, counts=True),
                    'bin_edges': bin_edges
                }

            # Compute percentage of spikes in refractory period
            refractory_period_index = np.logical_and(
                autocorrelations['refractory_period']['bin_edges'] > 0,
                autocorrelations['refractory_period']['bin_edges'] <= Params.refractory_period
            )[:-1]
            refractory_error_rate = (np.sum(autocorrelations['refractory_period']['values'][refractory_period_index])
                                     / timestamps.size)
            autocorrelations['refractory_period']['refractory_error_rate'] = refractory_error_rate

            for unit in recordings_unit:

                if unit is None:
                    continue

                unit['analysis']['autocorrelations'] = autocorrelations

    return recordings


def create_df_fields_for_recordings(recordings):

    recordings.df_fields = pd.DataFrame(
        columns=('animal', 'animal_field', 'animal_unit', 'experiment_id', 'stability_minutes', 'stability_halves',
                 'peak_spike_rate', 'median_spike_rate', 'peak_x', 'peak_y', 'area',
                 'peak_nearest_corner', 'peak_nearest_wall',
                 )
    )

    for animal_field_ind, field in enumerate(recordings[0].analysis['fields']):

        recordings.df_fields = recordings.df_fields.append(field['properties'], ignore_index=True)


def create_unit_data_frames_for_recordings(recordings):

    dataframe_columns = ('animal', 'animal_unit', 'tetrode_nr', 'tetrode_cluster_id',
                         'channel_group', 'isolation_distance', 'l_ratio', 'refractory_error_rate',
                         'category', 'max_stability_minutes', 'max_stability_halves', 'num_fields',
                         'amplitude', 'half_width', 'peak_to_trough', 'trough_ratio',
                         'pca_component_1', 'pca_component_2', 'pca_component_3')

    recordings.df_units = pd.DataFrame(columns=dataframe_columns)

    for i_recording_unit, recordings_unit in enumerate(recordings.units):

        max_stability_minutes = \
            np.nanmax([unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['stability']
                       for unit in recordings_unit if not (unit is None)])
        max_stability_halves = \
            np.nanmax([unit['analysis']['spatial_ratemaps']['spike_rates_halves']['stability']
                       for unit in recordings_unit if not (unit is None)])

        unit = recordings_unit[snippets.get_index_where_most_spikes_in_unit_list(recordings_unit)]
        i_max_amp = np.argmax(unit['analysis']['mean_waveform_properties']['amplitude'])

        recordings.df_units = recordings.df_units.append(
            {
                'animal': recordings[0].info['animal'],
                'animal_unit': i_recording_unit,
                'tetrode_nr': unit['tetrode_nr'],
                'tetrode_cluster_id': unit['tetrode_cluster_id'],
                'channel_group': unit['channel_group'],
                'mean_firing_rate': recordings.mean_firing_rate_of_unit(i_recording_unit),
                'isolation_distance': unit['analysis']['sorting_quality']['isolation_distance'],
                'l_ratio': unit['analysis']['sorting_quality']['l_ratio'],
                'refractory_error_rate':
                    unit['analysis']['autocorrelations']['refractory_period']['refractory_error_rate'],
                'category': unit['analysis']['category'],
                'max_stability_minutes': max_stability_minutes,
                'max_stability_halves': max_stability_halves,
                'num_fields': sum(recordings.df_fields['animal_unit'] == i_recording_unit),
                'amplitude': unit['analysis']['mean_waveform_properties']['amplitude'][i_max_amp],
                'half_width': unit['analysis']['mean_waveform_properties']['half_width'][i_max_amp],
                'peak_to_trough': unit['analysis']['mean_waveform_properties']['peak_to_trough'][i_max_amp],
                'trough_ratio': unit['analysis']['mean_waveform_properties']['trough_ratio'][i_max_amp],
                'pca_component_1': unit['analysis']['mean_waveform_properties']['pca_component_1'][i_max_amp],
                'pca_component_2': unit['analysis']['mean_waveform_properties']['pca_component_2'][i_max_amp],
                'pca_component_3': unit['analysis']['mean_waveform_properties']['pca_component_3'][i_max_amp]
            },
            ignore_index=True
        )


class PositionDecoding(object):

    @staticmethod
    def extract_information_from_posteriors_for_recordings(posteriors, recordings):

        # Get mappings from map bin indices to xy coordinates
        x_from_bin = []
        y_from_bin = []
        for j_recording in range(len(posteriors)):
            x_tmp, y_tmp = SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(
                posteriors[j_recording].shape[:2],
                (0, recordings[j_recording].info['arena_size'][0], 0, recordings[j_recording].info['arena_size'][1])
            )
            x_from_bin.append(x_tmp)
            y_from_bin.append(y_tmp)

        # Iterate over each timepoint
        peak_xy = []
        peak_recording_ind = []
        peak_value = []
        peak_value_normalised = []
        # peak_gauss_params = []
        for n_posterior in range(posteriors[0].shape[2]):

            # Find peak posterior value for this timepoint and its location and recording index
            peak_v_out = -1.0
            peak_l_out = None
            peak_r_ind = None
            for j_recording in range(len(posteriors)):

                peak_v = np.nanmax(posteriors[j_recording][:, :, n_posterior])
                if peak_v > peak_v_out:

                    peak_v_out = peak_v

                    if np.sum((posteriors[j_recording][:, :, n_posterior] == peak_v).flatten()) > 1:

                        # If multiple positions had the same value, pick one at random
                        inds = np.where((posteriors[j_recording][:, :, n_posterior] == peak_v).flatten())[0]
                        peak_l_out = np.unravel_index(inds[int(np.random.randint(len(inds)))],
                                                      posteriors[j_recording].shape[:2])

                    else:

                        peak_l_out = np.unravel_index(np.nanargmax(posteriors[j_recording][:, :, n_posterior]),
                                                      posteriors[j_recording].shape[:2])

                    peak_r_ind = j_recording

            if peak_v_out == 0.0:
                # If peak value was 0, set location ambiguous
                peak_r_ind = -1

            x = x_from_bin[peak_r_ind][int(peak_l_out[1])]
            y = y_from_bin[peak_r_ind][int(peak_l_out[0])]
            peak_xy.append([x, y])
            peak_recording_ind.append(peak_r_ind)
            peak_value.append(peak_v_out)

            # Compute peak_value_normalised
            posterior_sum = sum([np.nansum(p[:, :, n_posterior].flatten()) for p in posteriors])
            peak_value_normalised.append(peak_v_out / posterior_sum)

        return {'peak_xy': np.array(peak_xy),
                'peak_recording_ind': np.array(peak_recording_ind),
                'peak_value': np.array(peak_value),
                'peak_value_normalised': np.array(peak_value_normalised),
                'posterior_shapes': np.array([posterior.shape[:2] for posterior in posteriors])}

    @staticmethod
    def compute_decoding_results_for_with_set_of_parameters(recordings, i_recording=None, verbose=False,
                                                            name='position_decoding', **kwargs):

        # Iterate over each recording, computing posteriors for timepoints in those recordings
        # If i_recording was provided, only compute values for the specified recording
        for i_recording in (range(len(recordings)) if i_recording is None else [i_recording]):

            if verbose:
                print('Bayes decoding "{}" for recording {}/4'.format(name, i_recording + 1))

            decoder = FlatPriorBayesPositionDecoding(
                recordings,
                xy_masks=[recording.position['analysis']['ratemap_speed_mask'] for recording in recordings],
                normalise=False,
                single_environment=False,
                verbose=verbose,
                **kwargs
            )

            posteriors, position_inds = decoder.posteriors_for_recording(i_recording)

            # Iterate over all i posteriors and save them to the folder of i recording
            for j_recording in range(len(posteriors)):
                np.save(os.path.join(os.path.dirname(recordings[i_recording].fpath),
                                     'posteriors_{}.npy'.format(recordings[j_recording].info['experiment_id'])),
                        posteriors[j_recording])

            # Get and store peak information for each timepoint
            recordings[i_recording].analysis[name] = \
                PositionDecoding.extract_information_from_posteriors_for_recordings(posteriors, recordings)

            # Store position_inds to NWB file
            recordings[i_recording].analysis[name]['position_inds'] = np.array(position_inds)

    @staticmethod
    def preprocess_animal_position_decoding(fpath, animal_id, recompute=False, verbose=False, **kwargs):
        """Computes posterior probability matrices for the first four recordings in session with
        :py:class:`barrylab_ephys_analysis.models.decoding.FlatPriorBayesPositionDecoding` and stores
        the results in the respective folders of each recording, as well as the NWB file.

        :param str fpath: path to ExpScales folder that contains folders with animal_ids
        :param str animal_id: fpath sub-folder name containing the data of the animal
        :param bool recompute: if False (default), new analysis is only computed if not yet present
            in the first recording folder of the animal. Note, checking for presence of analysis is not perfect.
        :param bool verbose: if True, more details are printed out during processing (like progress bars)
        :param kwargs: any additional keyword arguments are passed to :py:class:`...recording_io.Recordings`
        """

        if verbose:
            print('Preprocessing bayes decoding for animal {}'.format(animal_id))

        fpaths = load.get_paths_to_animal_recordings_on_single_day(fpath, animal_id)[:4]

        if recompute:
            position_decoding_available = [False]
        else:
            # Check if data is available in file
            position_decoding_available = []
            for fpath in fpaths:
                position_decoding_available.append(
                    check_if_analysis_field_in_file(
                        fpath,
                        'position_decoding_large_window'
                    )
                )

        if all(position_decoding_available):

            return

        # Load the data

        recordings = Recordings(fpaths, clustering_name=Params.clustering_name, no_waveforms=True,
                                continuous_data_type=None, verbose=verbose, **kwargs)
        recordings.load_analysis()

        # Compute decoding results with default parameters
        PositionDecoding.compute_decoding_results_for_with_set_of_parameters(
            recordings, verbose=verbose, **deepcopy(Params.bayes_position_decoding)
        )

        # Save results to NWB file
        recordings.save_analysis()

        del recordings


def find_tetrode_with_most_units_in_each_channel_group(recordings):

    tetrodes_with_place_cells = []
    for i in range(len(recordings.units)):
        unit = recordings.first_available_recording_unit(i)
        if unit['analysis']['category'] == 'place_cell':
            tetrodes_with_place_cells.append(unit['tetrode_nr'])
    tetrodes_with_place_cells, place_cell_counts = np.unique(tetrodes_with_place_cells, return_counts=True)
    tetrode_areas = np.array(list(
        map(lambda tetrode_nr: recordings[0].tetrode_channel_group(tetrode_nr, recordings[0].channel_map),
            tetrodes_with_place_cells)
    ))

    max_place_cell_tetrode = {}
    for channel_group in recordings[0].channel_map:
        idx = tetrode_areas == channel_group
        max_place_cell_tetrode[channel_group] = \
            tetrodes_with_place_cells[idx][int(np.argmax(place_cell_counts[idx]))]

    return max_place_cell_tetrode


def preprocess_animal_lfp_spectral_overview(recordings=None, fpaths=None, recompute=False, verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        spectral_data_available = [False]
    else:
        # Check if data is available in file
        spectral_data_available = []
        for fpath in fpaths:
            spectral_data_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'spectral_overview'
                )
            )

    if all(spectral_data_available):
        return recordings

    if recordings is None:
        recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

    create_df_fields_for_recordings(recordings)

    if verbose:
        print('Computing spectral overview for animal {}'.format(recordings[0].info['animal']))

    # Find tetrodes with most place cells in both hemispheres
    max_place_cell_tetrode = find_tetrode_with_most_units_in_each_channel_group(recordings)

    # Compute power spectral density for the max place cell tetrode in each group for all recordings

    for recording in recordings:

        recording.analysis['spectral_overview'] = {}

        for channel_group in recording.channel_map:

            frequencies, psd = spectral.welch_psd(
                recording.continuous['continuous'][:, max_place_cell_tetrode[channel_group]],
                freq_res=0.1,
                fs=recording.continuous['sampling_rate']
            )

            recording.analysis['spectral_overview'][channel_group] = {
                'tetrode_nr': max_place_cell_tetrode[channel_group],
                'frequencies': frequencies,
                'psd': psd
            }

    return recordings


def preprocess_animal_theta(recordings=None, fpaths=None, recompute=False, verbose=True, **kwargs):

    if recordings is None and fpaths is None:
        raise ValueError('Either recordings or fpaths must be provided')

    if fpaths is None:
        fpaths = recordings.fpaths

    if recompute:
        data_available = [False]
    else:
        # Check if data is available in file
        data_available = []
        for fpath in fpaths:
            data_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'analog_signals/theta_amplitude'
                )
            )
            data_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'analog_signals/theta_phase'
                )
            )
            data_available.append(
                check_if_analysis_field_in_file(
                    fpath,
                    'analog_signals/theta_frequency'
                )
            )

    if all(data_available):
        return recordings

    if recordings is None:
        recordings = load_recordings_and_correct_experiment_ids(fpaths, verbose=verbose, **kwargs)

    if verbose:
        print('Computing theta analog signals for animal {}'.format(recordings[0].info['animal']))

    for recording in recordings:

        channel_group_tetrodes = {channel_group: item['tetrode_nr']
                                  for channel_group, item in recording.analysis['spectral_overview'].items()}

        theta_amplitude = FrequencyBandAmplitude(recording, 'theta_amplitude',
                                                 channel_group_tetrodes=channel_group_tetrodes, verbose=verbose,
                                                 **Params.theta_amplitude)

        theta_phase = FrequencyBandPhase(recording, 'theta_phase',
                                         channel_group_tetrodes=channel_group_tetrodes, verbose=verbose,
                                         **Params.theta_phase)

        theta_frequency = FrequencyBandFrequency(recording, 'theta_frequency',
                                                 channel_group_tetrodes=channel_group_tetrodes, verbose=verbose,
                                                 **Params.theta_frequency)

        analog_signals = (theta_amplitude, theta_phase, theta_frequency)

        for analog_signal in analog_signals:
            analog_signal.compute(recompute=True)
            analog_signal.add_to_recording()

    return recordings


def preprocess_animal(fpath, animal_id, recompute=False, save=True, verbose=False, **kwargs):
    """Performs pre-processing all recordings of a single (first) day found at fpath for animal_id.

    Some lazy hard-coded checking is done to identify whether the analysis already available.

    :param str fpath: path to ExpScales folder that contains folders with animal_ids
    :param str animal_id: fpath sub-folder name containing the data of the animal
    :param bool recompute: if False (default), new analysis is only computed if not yet present
        in NWB file. Note, checking for presence of analysis is not perfect.
    :param bool save: if True, analysis results are saved to the NWB file
    :param bool verbose: if True, more details are printed out during processing (like progress bars)
    :param kwargs: any additional keyword arguments are passed to :py:class:`...recording_io.Recordings`
    :return: Recordings instance with preprocessing complete
    """

    if verbose:
        print('Preprocessing data for animal {}'.format(animal_id))

    recordings = None

    fpaths = load.get_paths_to_animal_recordings_on_single_day(fpath, animal_id)

    recordings = compute_unit_autocorrelations_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )
    if save and not (recordings is None):
        if verbose:
            print('Saving intermediate results for animal {}'.format(animal_id))
        recordings.save_analysis()

    recordings = compute_waveform_properties_and_sorting_quality_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )
    if save and not (recordings is None):
        if verbose:
            print('Saving intermediate results for animal {}'.format(animal_id))
        recordings.save_analysis()

    recordings = compute_ratemap_speed_mask_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    recordings = compute_ratemaps_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    recordings = compute_ratemap_stability_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    if save and not (recordings is None):
        if verbose:
            print('Saving intermediate results for animal {}'.format(animal_id))
        recordings.save_analysis()

    recordings = detect_fields_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    recordings = assign_unit_categories_if_not_available(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    if save and not (recordings is None):
        if verbose:
            print('Saving intermediate results for animal {}'.format(animal_id))
        recordings.save_analysis()

    recordings = preprocess_animal_lfp_spectral_overview(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    recordings = preprocess_animal_theta(
        recordings=recordings, fpaths=fpaths, recompute=recompute, verbose=verbose,
        **kwargs
    )

    if save and not (recordings is None):
        if verbose:
            print('Saving intermediate results for animal {}'.format(animal_id))
        recordings.save_analysis()

    # Clear recordings from memory as it is reloaded in next processing steps
    del recordings

    PositionDecoding.preprocess_animal_position_decoding(
        fpath, animal_id, recompute=recompute, verbose=verbose, **kwargs
    )

    if verbose:
        print('Preprocessing complete for animal {}'.format(animal_id))


def preprocess_and_save_all_animals(fpath, recompute=False, verbose=False, **kwargs):

    processes = []

    process_kwargs = {'save': True}

    process_kwargs.update({'recompute': recompute,
                           'verbose': verbose})
    process_kwargs.update(kwargs)

    for animal_id in Params.animal_ids:
        p = Process(target=preprocess_animal, args=(fpath, animal_id), kwargs=process_kwargs)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def main():

    parser = argparse.ArgumentParser(description='Preprocess data for Experiment Scales analysis.')
    parser.add_argument('fpath', type=str, nargs=1,
                        help='path to ExpScales folder that contains folders with animal_ids.')
    parser.add_argument('--recompute', type=lambda x: eval(x) if len(x) == 4 else False, nargs=1,
                        help='True or False (default)',
                        default=[False])
    parser.add_argument('--verbose', type=lambda x: eval(x) if len(x) == 4 else False, nargs=1,
                        help='True or False (default)',
                        default=[False])
    parser.add_argument('--animal_id', type=str, nargs=1,
                        help='specify animal_id to only preprocess data for a single animal.')
    kwargs = argparse_to_kwargs(parser)

    if 'animal_id' in kwargs:
        preprocess_animal(**kwargs)
    else:
        preprocess_and_save_all_animals(**kwargs)


if __name__ == '__main__':
    main()
