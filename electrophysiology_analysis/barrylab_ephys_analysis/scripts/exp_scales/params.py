
import numpy as np


class Params(object):
    """
    This class contains all the parameters used throughout the analysis.
    This is to ensure all parameters used in the analysis can be viewed and edited conveniently.

    Any desired changes to the parameters of the analysis should be done here in class definition.

    The params class should not be changed dynamically by the code.
    """

    # Animals included in the analysis
    animal_ids = ('R2470', 'R2474', 'R2478', 'R2481', 'R2482')

    # Name for unit cluster identities to load
    clustering_name = 'manual_1'

    # Features used for computing sorting quality measures
    sorting_quality_feature_names = (
        'amplitude',
        'time_to_peak',
        'time_to_trough',
        'peak_to_trough',
        'half_width',
        'trough_ratio',
        'pca_component_1',
        'pca_component_2',
        'pca_component_3'
    )

    # Duplicate detection criteria
    duplicate_cell_criteria = {
        'max_lag': 0.025,
        'bin_size': 0.002,
        'coincident_spikes_peak_width_threshold': 0.0005,
        'coincidental_spike_count_threshold': 200,
        'ratemap_correlation_threshold': 0.5,
        'ratemap_min_spikes': 200,
        'category': 'place_cell'
    }

    # Computation of xy position mask
    xy_masking = {
        'speed_smoothing_window': 0.1,  # seconds
        'speed_smoothing_method': 'gaussian',
        'speed_threshold': 10  # cm/s
    }

    # Computation of single unit ratemap
    spatial_ratemap = {
        'bin_size': 4,
        'n_smoothing_bins': 2,
        'smoothing_method': 'gaussian',
        'adaptive_smoothing_alpha': 100
    }

    # Parameters for creating direction filtered ratemaps
    direction_filtered_spatial_ratemap = {
        'direction_bin_centers': np.arange(-np.pi, np.pi, np.pi / 4),
        'direction_bin_width': np.pi / 2
    }

    directional_ratemap = {
        'bin_size': np.pi / 30.,
        'n_smoothing_bins': 2,
        'smoothing_method': 'gaussian'
    }

    # Autocorrelation computation
    autocorrelation_params = {
        'refractory_period': {
            'bin_size': 0.001,
            'max_lag': 0.05,
        },
        'theta_modulation': {
            'bin_size': 0.01,
            'max_lag': 0.5
        }
    }

    # Refractory error rate calculation
    refractory_period = 0.0025  # minimum allowed latency between spikes to not count as refractory error

    # Location directory name for analysis output in root directory of the dataset
    analysis_path = 'Analysis'

    # Used for quantifying ratemap stability with spatial correlation
    ratemap_stability_kwargs = {
        'min_included_value': 0.01,  # Also used for thesis figure Remapping
        'min_bins': 6
    }

    # Detecting fields from single unit ratemaps
    field_detection = {
        'base_threshold': 1.0,
        'threshold_step': 0.05,  # field detection threshold shift per iteration
        'primary_filter_kwargs': {'min_area_bins': 10,
                                  'min_peak_value': 2},
        'secondary_filter_kwargs': {'min_stability': 0.25,
                                    'max_relative_bins': 0.5,
                                    'stability_kwargs': ratemap_stability_kwargs}
    }

    # tetrode_cluster_id value that identifies noise (combined unsorted spike events)
    noise_tetrode_cluster_id = 1

    # Place Cell categorisation
    place_cell_criteria = {
        'min_peak_to_trough': 450e-6,
        'min_half_width': 100e-6,
        'min_stability_minutes': 0.5,
        'min_stability_halves': 0.25,
        'min_number_of_fields': 1,
        'max_mean_firing_rate': 4,
        'min_trough_ratio': 0.175
    }

    # Interneuron categorisation
    interneuron_criteria = {
        'max_half_width': 150e-6,
        'max_stability_halves': 0.75,
        'min_mean_firing_rate': 4,
        'max_trough_ratio': 0.4
    }

    # Flat Prior Bayes decoding parameters
    bayes_position_decoding = {
        'name': 'position_decoding_large_window',
        'decoding_window_size': 31,  # number of position samples
        'decoding_window_overlap': 30,  # number of position samples
        'category': 'place_cell',
        'ratemap_kwargs': {
            'bin_size': 4,
            'n_smoothing_bins': 2,
            'smoothing_method': 'gaussian'
        },
        'cv_segment_size': 30 * 60 * 5,  # number of position samples
    }

    # Theta frequency hilbert computation
    theta_amplitude = {
        'highpass_frequency': 7.0,
        'lowpass_frequency': 11.0,
        'filter_order': 2,
        'temporal_smoothing_sigma': 0.1,
    }

    # Theta frequency hilbert computation
    theta_phase = {
        'highpass_frequency': 7.0,
        'lowpass_frequency': 11.0,
        'filter_order': 2,
        'temporal_smoothing_sigma': 0.005,
    }

    # Theta frequency hilbert computation
    theta_frequency = {
        'highpass_frequency': 7.0,
        'lowpass_frequency': 11.0,
        'filter_order': 2,
        'temporal_smoothing_sigma': 0.1,
    }
