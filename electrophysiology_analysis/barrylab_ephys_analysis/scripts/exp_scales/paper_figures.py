import os
import pickle
import sys
from random import shuffle
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, pearsonr, linregress, friedmanchisquare, gamma, poisson, zscore
from scipy.special import gamma as gamma_function
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy import ndimage
import pandas as pd
from tqdm import tqdm
from pingouin import partial_corr
import matplotlib


matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import seaborn as sns

from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap
from barrylab_ephys_analysis.spatial.similarity import spatial_correlation
from barrylab_ephys_analysis.scripts.exp_scales import snippets
from barrylab_ephys_analysis.spikes.correlograms import plot_correlogram
from barrylab_ephys_analysis.scripts.exp_scales.params import Params

from barrylab_ephys_analysis.recording_io import Recording
from barrylab_ephys_analysis.scripts.exp_scales import load
from barrylab_ephys_analysis.scripts.exp_scales.paper_preprocess import preprocess_and_save_all_animals, \
    create_df_fields_for_recordings, create_unit_data_frames_for_recordings
from barrylab_ephys_analysis.spatial.fields import compute_field_contour
from barrylab_ephys_analysis.spikes.utils import count_spikes_in_sample_bins, convert_spike_times_to_sample_indices
from barrylab_ephys_analysis.lfp.oscillations import FrequencyBandFrequency


from barrylab_ephys_analysis.scripts.exp_scales.paper_methods import ValueByBinnedDistancePlot, \
    plot_raincloud_and_stats, get_max_ylim, compute_pairwise_comparisons, plot_stats_dict_to_axes, \
    SpatialFilteringLegend, get_field_data_with_distance_to_boundary, FieldCoverage, RatemapsPlot, \
    BayesianPositionDecodingArenaAccuracy, compute_distances_to_landmarks, filter_dataframe_by_direction, \
    PlaceFieldPeakDistribution, PopulationVectorChangeRate


seaborn_font_scale = 1.5

sns_colors = sns.color_palette(n_colors=(5 + 5 + 4))

sns_animal_colors = sns_colors[:5]
sns_environment_colors = sns_colors[5:10]
sns_other_colors = sns_colors[10:]


def paper_figures_path(fpath):
    path = os.path.join(fpath, Params.analysis_path, 'PaperFigures')
    if not os.path.isdir(path):
        os.mkdir(path)

    return path


spatial_filter_legend_instance = SpatialFilteringLegend()


main_experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')


experiment_id_substitutes = {
    'exp_scales_a': 'A',
    'exp_scales_b': 'B',
    'exp_scales_c': 'C',
    'exp_scales_d': 'D',
    'exp_scales_a2': "A'"
}


experiment_id_substitutes_inverse = {value: key for key, value in experiment_id_substitutes.items()}


spatial_windows = {
    'exp_scales_a': (0, 87.5, 0, 125),
    'exp_scales_b': (0, 175, 0, 125),
    'exp_scales_c': (0, 175, 0, 250),
    'exp_scales_d': (0, 350, 0, 250),
    'exp_scales_a2': (0, 87.5, 0, 125)
}

arena_areas_meters = {
    'exp_scales_a': 87.5 * 125 / 10000,
    'exp_scales_b': 175 * 125 / 10000,
    'exp_scales_c': 175 * 250 / 10000,
    'exp_scales_d': 350 * 250 / 10000,
    'exp_scales_a2': 87.5 * 125 / 10000
}


arena_areas_meters_short_env = {experiment_id_substitutes[key]: value for key, value in arena_areas_meters.items()}


experiment_ids_with_areas = {
    'exp_scales_a': 'A, {:.2f}'.format(arena_areas_meters['exp_scales_a']),
    'exp_scales_b': 'B, {:.2f}'.format(arena_areas_meters['exp_scales_b']),
    'exp_scales_c': 'C, {:.2f}'.format(arena_areas_meters['exp_scales_c']),
    'exp_scales_d': 'D, {:.2f}'.format(arena_areas_meters['exp_scales_d']),
    'exp_scales_a2': "A', {:.2f}".format(arena_areas_meters['exp_scales_a2'])
}


experiment_ids_with_areas_ticks = {
    'ticks': [0.0] + [arena_areas_meters[x] for x in main_experiment_ids],
    'ticklabels': ['0'] + [experiment_ids_with_areas[x] for x in main_experiment_ids]
}


def construct_df_population_vector_change_file_path(fpath):
    return os.path.join(fpath, Params.analysis_path, 'df_population_vector_change.p')


class ExampleUnit(object):

    @staticmethod
    def make_ratemaps_subfigure(recordings, recordings_unit_ind):
        fig = RatemapsPlot.make_default_figure()
        RatemapsPlot.plot(recordings, recordings_unit_ind, fig, draw_gaussians=False, draw_ellipses=False)
        return fig

    @staticmethod
    def plot_waveforms(recordings, recordings_unit_ind, fig, gs, time_bar_ms=0.5, volt_bar_uv=100):
        """Plots waveforms of the unit from all 4 channels.

        The time and voltage scale bars are in size specified in input arguments.
        """

        index = snippets.get_index_where_most_spikes_in_unit_list(recordings.units[recordings_unit_ind])
        waveforms = recordings.units[recordings_unit_ind][index]['analysis']['mean_waveform']
        sampling_rate = recordings.units[recordings_unit_ind][index]['sampling_rate']

        y_limits = [np.amin(-waveforms) * 1.02, np.amax(-waveforms) * 1.02]
        for nw, gs_field in zip(range(4), gs):

            ax = fig.add_subplot(gs_field)
            ax.plot(-waveforms[nw, :])
            ax.set_ylim(y_limits)
            ax.axis('off')

            if nw == 2:
                time_bar_length = (time_bar_ms / 1000) * sampling_rate
                x_right_edge = ax.get_xlim()[1]
                top_edge = y_limits[1] * 0.95
                ax.plot([x_right_edge - time_bar_length, x_right_edge], [top_edge, top_edge],
                        color='black', linewidth=2.5)
                ax.plot([x_right_edge, x_right_edge], [top_edge - volt_bar_uv, top_edge],
                        color='black', linewidth=2.5)

    @staticmethod
    def plot_autocorrelograms(recordings, recordings_unit_ind, fig, gs):
        """Autocorrelation plot parameters are specified in Params.autocorrelation_params.

        These should be the following:
            top plot:    'bin_size': 0.001 seconds, 'max_lag': 0.05 seconds,
            bottom plot: 'bin_size': 0.01 seconds, 'max_lag': 0.5 seconds,
        """

        index = snippets.get_index_where_most_spikes_in_unit_list(recordings.units[recordings_unit_ind])
        autocorrelations = recordings.units[recordings_unit_ind][index]['analysis']['autocorrelations']

        for correlation_name, gs_field in zip(sorted(autocorrelations), gs):
            ax = fig.add_subplot(gs_field)
            ax.axis('off')
            plot_correlogram(autocorrelations[correlation_name]['values'],
                             autocorrelations[correlation_name]['bin_edges'],
                             ax)

    @staticmethod
    def make_waveform_and_autocorrelogram_subfigure(recordings, recordings_unit_ind):

        fig = plt.figure(figsize=(RatemapsPlot.default_figure_size[0], RatemapsPlot.default_figure_size[1] / 3.))
        gs_main = fig.add_gridspec(1, 2)
        gs_main.tight_layout(fig, pad=0.2)
        gs_waveforms = GridSpecFromSubplotSpec(2, 2, gs_main[0], wspace=0, hspace=0)
        gs_correlograms = GridSpecFromSubplotSpec(2, 1, gs_main[1])

        ExampleUnit.plot_waveforms(recordings, recordings_unit_ind, fig, gs_waveforms)
        ExampleUnit.plot_autocorrelograms(recordings, recordings_unit_ind, fig, gs_correlograms)

        return fig

    @staticmethod
    def write(fpath, all_recordings, df_units, unit=20, prefix='', verbose=True):

        figure_name = prefix + 'ExampleUnit'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        recordings = [x for x in all_recordings if x[0].info['animal'] == df_units.loc[unit, 'animal']][0]
        recordings_unit_ind = df_units.loc[unit, 'animal_unit']

        fig = ExampleUnit.make_waveform_and_autocorrelogram_subfigure(recordings, recordings_unit_ind)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}_waveforms.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}_waveforms.svg'.format(figure_name)))
        plt.close(fig)

        fig = ExampleUnit.make_ratemaps_subfigure(recordings, recordings_unit_ind)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}_ratemaps.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}_ratemaps.svg'.format(figure_name)))
        plt.close(fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldDetectionMethod(object):

    @staticmethod
    def plot(all_recordings, df_units, axs, unit=1136, experiment_id='exp_scales_c'):

        recordings = [x for x in all_recordings if x[0].info['animal'] == df_units.loc[unit, 'animal']][0]
        recordings_unit_ind = df_units.loc[unit, 'animal_unit']
        i_recording = [i for i in range(len(recordings)) if recordings[i].info['experiment_id'] == experiment_id][0]
        unit = recordings.units[recordings_unit_ind][i_recording]
        ratemap = unit['analysis']['spatial_ratemaps']['spike_rates_smoothed']

        # Plot ratemap
        axes_image = axs[0].imshow(ratemap, cmap='jet')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size='{}%'.format(10), pad=0.05)
        axs[0].figure.colorbar(axes_image, cax=cax, ax=axs[0])
        cax.set_ylabel('spike rate (Hz)')

        field_inds = np.where((recordings.df_fields['experiment_id'] == experiment_id)
                              & (recordings.df_fields['animal_unit'] == recordings_unit_ind))[0]
        contours = [compute_field_contour(recordings[0].analysis['fields'][i]['ratemap']) for i in field_inds]

        colors = [np.array([1, 0.64, 0]), (np.array((165, 42, 42)) / 255)]

        for contour, color in zip(contours, colors):
            SpatialRatemap.plot_contours(contour, axs[0], color=color)

        # Create an RGB array
        array = np.ones((ratemap.shape[0], ratemap.shape[1], 3), dtype=np.float32)
        array[:, np.array([0, ratemap.shape[1] - 1])] = 0
        array[np.array([0, ratemap.shape[0] - 1]), :] = 0

        # Plot first threshold
        threshold = 1
        field_map = ndimage.label(ratemap > threshold)[0]
        inds = np.where(field_map == 1)
        array[inds[0], inds[1], :] = np.array([0, 0.5, 0])[None, None, :]
        axs[1].imshow(array)
        axs[1].set_title('threshold = {0:#.3g} Hz'.format(threshold))

        # Plot second threshold
        threshold = 1.25
        field_map = ndimage.label(ratemap > threshold)[0]
        inds = np.where(field_map == 1)
        array[inds[0], inds[1], :] = colors[1][None, None, :]  # Brown
        inds = np.where(field_map == 2)
        array[inds[0], inds[1], :] = colors[0][None, None, :]
        axs[2].imshow(array)
        axs[2].set_title('threshold = {0:#.3g} Hz'.format(threshold))

        # Plot third threshold
        threshold = 4.75
        field_map = ndimage.label(ratemap > threshold)[0]
        inds = np.where(field_map == 1)
        array[inds[0], inds[1], :] = np.array([1, 0.75, 0.79])[None, None, :]
        inds = np.where(field_map == 2)
        array[inds[0], inds[1], :] = np.array([1, 0, 0])[None, None, :]
        axs[3].imshow(array)
        axs[3].set_title('threshold = {0:#.3g} Hz'.format(threshold))

    @staticmethod
    def make_figure(all_recordings, df_units):

        fig, axs = plt.subplots(1, 4, figsize=(12, 4), gridspec_kw={'width_ratios': (1.4, 1, 1, 1)})
        plt.subplots_adjust(left=0, right=0.995, bottom=0.15, top=0.85, wspace=0.3)

        for ax in axs:
            ax.axis('off')

        FieldDetectionMethod.plot(all_recordings, df_units, axs)

        return fig

    @staticmethod
    def write(fpath, all_recordings, df_units, prefix='', verbose=True):

        figure_name = prefix + 'FieldDetectionMethod'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig = FieldDetectionMethod.make_figure(all_recordings, df_units)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        plt.close(fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class IntraTrialCorrelations(object):

    @staticmethod
    def compute(all_recordings):

        per_unit_animal = []
        per_unit_environment = []
        per_unit_minutes = []
        per_unit_halves = []

        for recordings in all_recordings:

            for i_recording, recording in enumerate(recordings[:4]):

                if not (recording.info['experiment_id'] in main_experiment_ids):
                    continue

                odd_minute_ratemap_stack = []
                even_minute_ratemap_stack = []

                first_half_ratemap_stack = []
                second_half_ratemap_stack = []

                # Get correlations of each unit and collect ratemap stacks

                for unit in recording.units:
                    if unit['analysis']['category'] != 'place_cell':
                        continue

                    # Compute per unit correlations

                    per_unit_animal.append(recording.info['animal'])
                    per_unit_environment.append(experiment_id_substitutes[recording.info['experiment_id']])

                    per_unit_minutes.append(
                        spatial_correlation(
                            unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'],
                            unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even'],
                            **Params.ratemap_stability_kwargs
                        )[0]
                    )

                    per_unit_halves.append(
                        spatial_correlation(
                            unit['analysis']['spatial_ratemaps']['spike_rates_halves']['first'],
                            unit['analysis']['spatial_ratemaps']['spike_rates_halves']['second'],
                            **Params.ratemap_stability_kwargs
                        )[0]
                    )

                    # Collect ratemaps to stack

                    odd_minute_ratemap_stack.append(
                        unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'])
                    even_minute_ratemap_stack.append(
                        unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even'])
                    first_half_ratemap_stack.append(
                        unit['analysis']['spatial_ratemaps']['spike_rates_halves']['first'])
                    second_half_ratemap_stack.append(
                        unit['analysis']['spatial_ratemaps']['spike_rates_halves']['second'])

        # Create DataFrame

        df_per_unit = pd.DataFrame({
            'animal': per_unit_animal,
            'environment': per_unit_environment,
            'minutes': per_unit_minutes,
            'halves': per_unit_halves
        })

        return df_per_unit

    @staticmethod
    def plot(all_recordings, ax, stat_ax, stripplot_size=1):

        df = IntraTrialCorrelations.compute(all_recordings).rename(columns={'halves': 'Pearson $\it{r}$'})

        plot_raincloud_and_stats('environment', 'Pearson $\it{r}$', df, ax, stat_ax,
                                 palette=sns.color_palette(sns_environment_colors[:len(main_experiment_ids)]),
                                 x_order=[experiment_id_substitutes[experiment_id]
                                          for experiment_id in main_experiment_ids],
                                 stripplot_size=stripplot_size)
        ax.set_yticks([y for y in ax.get_yticks() if y <= 1])
        ax.set_xlabel('environment')

    @staticmethod
    def make_figure(all_recordings):

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.98)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(8, 8))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        IntraTrialCorrelations.plot(all_recordings, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def print_correlation_between_spatial_correlation_and_field_count_per_cell(all_recordings, df_units, df_fields):

        df = df_fields.loc[df_fields['experiment_id'] == 'exp_scales_d',
                           ['animal', 'animal_unit']].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df.loc[df['category'] == 'place_cell', ['animal', 'animal_unit']]  # Only keep place cell fields
        df['count'] = 1
        df = df.groupby(['animal', 'animal_unit'])['count'].sum().reset_index()

        animal_recordings = {recordings[0].info['animal']: recordings for recordings in all_recordings}
        animal_exp_scales_d_recording_index = {}
        for animal, recordings in animal_recordings.items():
            animal_exp_scales_d_recording_index[animal] = [
                recording.info['experiment_id'] for recording in animal_recordings[animal]
            ].index('exp_scales_d')

        spatial_correlations = []
        for animal, animal_unit in zip(df['animal'], df['animal_unit']):
            unit = animal_recordings[animal].units[animal_unit][animal_exp_scales_d_recording_index[animal]]
            spatial_correlations.append(
                unit['analysis']['spatial_ratemaps']['spike_rates_halves']['stability']
            )

        df['spatial_correlation_between_halves'] = spatial_correlations

        print()
        print('Correlation between spatial correlation (1st and 2nd half) and count of place fields in environment D: '
              'r={:.3f} p={:.6f} N={}'.format(
            *pearsonr(df['spatial_correlation_between_halves'], df['count']), df.shape[0]
        ))
        print()

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'IntraTrialCorrelations'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = IntraTrialCorrelations.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)

        IntraTrialCorrelations.print_correlation_between_spatial_correlation_and_field_count_per_cell(
            all_recordings, df_units, df_fields
        )

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class PlaceCellAndFieldCounts(object):

    @staticmethod
    def plot_stacked_bars(ax, df, value_name, legend=True):

        bar_bottom_heights = {environment: 0 for environment in df['environment'].unique()}
        colors = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}
        bars = {}
        for animal in sorted(df['animal'].unique()):
            df_tmp = df.loc[df['animal'] == animal].copy().sort_values('environment')

            bars[animal] = ax.bar(
                np.arange(len(bar_bottom_heights)),
                df_tmp[value_name],
                bottom=[bar_bottom_heights[environment] for environment in df_tmp['environment']],
                color=colors[animal],
                width=0.8,
                linewidth=0
            )

            for environment, value in zip(df_tmp['environment'], df_tmp[value_name]):
                bar_bottom_heights[environment] += value

        if legend:
            ax.legend(list(bars.values()), ['animal {}'.format(i) for i in range(1, len(bars) + 1)])

        ax.set_xticks(np.arange(len(bar_bottom_heights)))
        ax.set_xticklabels(sorted(list(bar_bottom_heights.keys())))
        ax.set_xlabel('environment')
        ax.set_ylabel(value_name)

    @staticmethod
    def plot_line_for_each_animal(ax, df, value_name, legend=True):

        df['environment_size'] = \
            np.array([FieldsDetectedAcrossEnvironments.environment_sizes[
                          FieldsDetectedAcrossEnvironments.environment_names.index(x)
                      ]
                      for x in df['environment']])

        colors = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}
        for i_animal, animal in enumerate(sorted(df['animal'].unique())):
            df_tmp = df.loc[df['animal'] == animal].copy().sort_values('environment')
            ax.plot(df_tmp['environment_size'], df_tmp[value_name], color=colors[animal],
                    label='{}'.format(i_animal + 1), marker='o', linewidth=2)

        if legend:
            ax.legend(title='animal')

        ax.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax.set_xticklabels(experiment_ids_with_areas_ticks['ticklabels'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('environment, size (m$^2$)')
        ax.set_ylabel(value_name)

    @staticmethod
    def plot_place_cell_count_in_environments(df_units, df_fields, ax, stat_ax):
        """The plot shows the count of place cells that have at least one field in each of the environments.
        """

        df = df_fields[['animal', 'animal_unit', 'experiment_id']].copy(deep=True)
        df.drop_duplicates(['animal', 'animal_unit', 'experiment_id'], inplace=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[df['category'] == 'place_cell'][['animal', 'experiment_id', 'animal_unit']]

        animal_place_cell_counts = {
            animal: df.loc[df['animal'] == animal, 'animal_unit'].unique().size
            for animal in df['animal'].unique()
        }

        df['place cell count'] = 1
        df = df[['animal', 'experiment_id', 'place cell count']]\
            .groupby(['animal', 'experiment_id']).sum().reset_index()

        df = df.loc[df['experiment_id'] != 'exp_scales_a2'].copy().reset_index()

        df.replace(to_replace={'experiment_id': experiment_id_substitutes}, inplace=True)
        df.rename(columns={'experiment_id': 'environment'}, inplace=True)

        PlaceCellAndFieldCounts.plot_line_for_each_animal(ax, df, 'place cell count', legend=False)

        table_cell_text = [['Animal', 'Cell count']]
        for animal, count in animal_place_cell_counts.items():
            table_cell_text.append([animal, '{:d}'.format(count)])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def plot_place_field_count_in_environments(df_units, df_fields, ax):
        """The plot shows the count of place fields that each animal had in each environment, across all place cells.
        """

        df = df_fields[['animal', 'animal_unit', 'experiment_id']].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[df['category'] == 'place_cell'][['animal', 'experiment_id']]

        df['place field count'] = 1
        df = df.groupby(['animal', 'experiment_id']).sum().reset_index()

        df = df.loc[df['experiment_id'] != 'exp_scales_a2'].copy().reset_index()

        df.replace(to_replace={'experiment_id': experiment_id_substitutes}, inplace=True)
        df.rename(columns={'experiment_id': 'environment'}, inplace=True)

        PlaceCellAndFieldCounts.plot_line_for_each_animal(ax, df, 'place field count', legend=True)

    @staticmethod
    def make_figure(df_units, df_fields):

        fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        plt.subplots_adjust(left=0.11, bottom=0.27, right=0.99, top=0.90, wspace=0.35)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        PlaceCellAndFieldCounts.plot_place_cell_count_in_environments(df_units, df_fields, axs[0], stat_ax)
        PlaceCellAndFieldCounts.plot_place_field_count_in_environments(df_units, df_fields, axs[1])

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'PlaceCellAndFieldCounts'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = PlaceCellAndFieldCounts.make_figure(df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldsDetectedAcrossEnvironments(object):
    environment_sizes = [0.875 * 1.25, 1.75 * 1.25, 1.75 * 2.5, 3.5 * 2.5]
    environment_names = ['A', 'B', 'C', 'D']

    @staticmethod
    def get_field_or_cell_count_per_environment_and_animal(df_units, df_fields):

        # Take care not to modify df_fields and only take relevant fields
        df_fields = df_fields.loc[df_fields['experiment_id'] != 'exp_scales_a2',
                                  ['unit', 'animal', 'experiment_id']].copy(deep=True)

        # Only keep fields for cell category specified
        df_fields = df_fields[df_fields['unit'].isin(np.where(df_units['category'] == 'place_cell')[0])]

        # Get total number of units per animal
        total_number_of_units = {}
        for animal in df_fields['animal'].unique():
            total_number_of_units[animal] = df_fields.loc[df_fields['animal'] == animal, 'unit'].unique().size

        # Get total number of fields per animal
        total_number_of_fields = {}
        for animal in df_fields['animal'].unique():
            total_number_of_fields[animal] = df_fields.loc[df_fields['animal'] == animal, 'unit'].size

        # Drop fields in exp_scales_a2
        df_fields = df_fields[df_fields['experiment_id'] != 'exp_scales_a2']

        # Replace experiment_id values for plotting and rename the column
        df_fields.replace(to_replace={'experiment_id': experiment_id_substitutes},
                          inplace=True)
        df_fields.rename(columns={'experiment_id': 'environment'}, inplace=True)

        # Count the number of units and fields present in each environment for each animal
        number_of_units = df_fields.drop_duplicates().groupby(['animal', 'environment']).count()
        number_of_units.reset_index(inplace=True)
        number_of_units.rename(columns={'unit': 'place cells'}, inplace=True)
        number_of_fields = df_fields.groupby(['animal', 'environment']).count()
        number_of_fields.reset_index(inplace=True)
        number_of_fields.rename(columns={'unit': 'place fields'}, inplace=True)

        # Compute percentage of total units and fields
        for animal, total in total_number_of_units.items():
            number_of_units.loc[number_of_units['animal'] == animal, 'place cells'] = \
                (number_of_units.loc[number_of_units['animal'] == animal, 'place cells'] / float(total))
        for animal, total in total_number_of_fields.items():
            number_of_fields.loc[number_of_fields['animal'] == animal, 'place fields'] = \
                (number_of_fields.loc[number_of_fields['animal'] == animal, 'place fields']
                 / float(total))

        # Set environment column equal to relative size to facilitate plotting
        number_of_units['environment_size'] = \
            np.array([FieldsDetectedAcrossEnvironments.environment_sizes[
                          FieldsDetectedAcrossEnvironments.environment_names.index(x)
                      ]
                      for x in number_of_units['environment']])
        number_of_fields['environment_size'] = \
            np.array([FieldsDetectedAcrossEnvironments.environment_sizes[
                          FieldsDetectedAcrossEnvironments.environment_names.index(x)
                      ]
                      for x in number_of_fields['environment']])

        # Compute area normalized percentage
        number_of_fields['place fields per unit area'] = \
            number_of_fields['place fields'].divide(number_of_fields['environment_size'])

        return number_of_units, number_of_fields

    @staticmethod
    def environment_field_density_model(area, slope, intercept):
        return (area * slope + intercept) / area

    @staticmethod
    def environment_field_density_proportional_to_baseline(area, slope, intercept):
        return FieldsDetectedAcrossEnvironments.environment_field_density_model(area, slope, intercept) / slope

    @staticmethod
    def compute_environment_area_with_field_density_correction(area, parameters):
        field_density_multiplier = \
            FieldsDetectedAcrossEnvironments.environment_field_density_proportional_to_baseline(
                area, parameters['slope'], parameters['intercept']
            )
        return area * field_density_multiplier

    @staticmethod
    def compute_environment_areas_with_field_density_correction(parameters=None):
        areas = {}
        for experiment_id in main_experiment_ids:
            if parameters is None:
                areas[experiment_id_substitutes[experiment_id]] = arena_areas_meters[experiment_id]
            else:
                areas[experiment_id_substitutes[experiment_id]] = \
                    FieldsDetectedAcrossEnvironments.compute_environment_area_with_field_density_correction(
                        arena_areas_meters[experiment_id], parameters
                    )

        areas['combined environment'] = np.sum(list(areas.values()))

        return areas

    @staticmethod
    def plot_place_field_distribution(df_units, df_fields, ax, stat_ax):
        """Shows the distribution of place fields of each animal between the different environments.
        Therefore, the values are a percentage of the total number of fields in a given animal.

        The dotted black line shows the least squares linear regression line.

        The inset shows the percentage of place fields in each square metre in each environment.
        These values are computed by dividing the values in main axes with the size of the environment.
        The x-axis units of inset axes are the same as main axes.

        Box plots show median, Q1, Q3 and range. As here N=5, each box plot element corresponds to
        one of the data points.
        """
        fit_plot_x_vals = np.linspace(1, 9, 100)

        _, df = \
            FieldsDetectedAcrossEnvironments.get_field_or_cell_count_per_environment_and_animal(
                df_units, df_fields
            )
        df = df.copy(deep=True)

        environments = sorted(np.unique(df['environment']))
        environment_sizes = sorted(np.unique(df['environment_size']))

        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        ax.scatter(df['environment_size'] + np.random.uniform(-0.2, 0.2, df['environment_size'].size),
                   df['place fields'],
                   s=50,
                   c=[colors_dict[x] for x in df['animal']],
                   linewidth=1, edgecolors='black', zorder=1, alpha=0.75)

        # Fit and plot linear model to distribution data

        main_line_slope, main_line_intercept, main_line_r_value, main_line_p_value, main_line_std_err = \
            linregress(df['environment_size'], df['place fields'])
        ax.plot(fit_plot_x_vals, main_line_intercept + main_line_slope * fit_plot_x_vals,
                color='black', linestyle=':', zorder=2)
        ax_text_right_side_x = fit_plot_x_vals[3 * len(fit_plot_x_vals) // 4]
        ax_text_right_side_y = main_line_intercept + main_line_slope * ax_text_right_side_x
        line_residuals = np.abs((main_line_intercept + main_line_slope * df['environment_size'])
                                - df['place fields'])
        main_line_mean_squared_error = np.mean(line_residuals ** 2)

        pearson_r_value, pearson_p_value = pearsonr(df['environment_size'], df['place fields'])

        # Plot linear model r value
        line_text = (
            'y = {:.3f}x + {:.3f}\n'.format(main_line_slope, main_line_intercept)
            + '$\it{r}$' + ' = {:.{prec}f}'.format(main_line_r_value, prec=2)
        )
        ax.text(ax_text_right_side_x, ax_text_right_side_y, line_text, ha='right', va='bottom')

        # Plot place field density

        ax_inset_height = \
            (main_line_intercept + main_line_slope * environment_sizes[-1] * 0.5) / ax.get_ylim()[1] * 100 * 0.75
        ax_inset = inset_axes(ax, width='45%', height='{:.0f}%'.format(ax_inset_height), loc='lower right')
        ax_inset.xaxis.tick_top()
        ax_inset.scatter(df['environment_size'] + np.random.uniform(-0.2, 0.2, df['environment_size'].size),
                         df['place fields per unit area'], s=50, c=[colors_dict[x] for x in df['animal']],
                         linewidth=1, edgecolors='black', zorder=1, alpha=0.75)

        ax_inset.plot(
            fit_plot_x_vals, (main_line_intercept + main_line_slope * fit_plot_x_vals) / fit_plot_x_vals,
            color='black', linestyle=':', zorder=2
        )

        # Adjust axes parameters

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_xlim((0, ax.get_xlim()[1]))
        ax.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax.set_xticklabels(experiment_ids_with_areas_ticks['ticklabels'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('environment, size (m$^2$)')
        ax.set_ylabel('proportion of place fields')

        ax_inset.set_ylim((0, ax_inset.get_ylim()[1]))
        ax_inset.set_xlim((0, ax_inset.get_xlim()[1]))
        ax_inset.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax_inset.set_xlabel('size (m$^2$)')
        ax_inset.set_ylabel('proportion\nof fields / m$^2$')
        plt.setp(ax_inset.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
        ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_inset.xaxis.set_label_position('top')
        ax_inset.xaxis.set_ticks_position('top')
        ax_inset.yaxis.labelpad = 0

        # Compute stats

        kruskal_h_value, kruskal_pvalue = \
            kruskal(*[df[df['environment'] == group]['place fields'] for group in environments])

        density_kruskal_h_value, density_kruskal_pvalue = \
            kruskal(*[df[df['environment'] == group]['place fields per unit area'] for group in environments])

        df_sorted = df.sort_values('animal')

        friedman_chisq_value, friedman_pvalue = \
            friedmanchisquare(*[df_sorted[df_sorted['environment'] == group]['place fields']
                                for group in environments])

        density_friedman_chisq_value, density_friedman_pvalue = \
            friedmanchisquare(*[df_sorted[df_sorted['environment'] == group]['place fields per unit area']
                                for group in environments])

        # Plot stats to stat_ax

        stat_ax.set_title('Place field formation')
        table_cell_text = [['Field distribution', 'H-value', 'p-value'],
                           ['Kruskal-Wallis test',
                            '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
                           ['', '', ''],
                           ['Field distribution', 'chi-square statistic', 'p-value'],
                           ['Friedman test',
                            '{:.2e}'.format(friedman_chisq_value), '{:.2e}'.format(friedman_pvalue)],
                           ['', '', ''],
                           ['Field density', 'H-value', 'p-value'],
                           ['Kruskal-Wallis test',
                            '{:.2e}'.format(density_kruskal_h_value), '{:.2e}'.format(density_kruskal_pvalue)],
                           ['', '', ''],
                           ['Field density', 'chi-square statistic', 'p-value'],
                           ['Friedman test',
                            '{:.2e}'.format(density_friedman_chisq_value), '{:.2e}'.format(density_friedman_pvalue)],
                           ['', '', ''],
                           ['fitted main linear model', 'parameters', ''],
                           ['', 'line_slope', '{:.3f}'.format(main_line_slope)],
                           ['', 'line_intercept', '{:.3f}'.format(main_line_intercept)],
                           ['', 'line_r_value', '{:.3f}'.format(main_line_r_value)],
                           ['', 'line_p_value', '{:.3e}'.format(main_line_p_value)],
                           ['', 'line_std_err', '{:.5f}'.format(main_line_std_err)],
                           ['', 'line_mean_squared_error', '{:.3e}'.format(main_line_mean_squared_error)],
                           ['', '', ''],
                           ['pearson', 'r', 'p'],
                           ['', '{:.3f}'.format(pearson_r_value), '{:.3e}'.format(pearson_p_value)],
                           ['', '', ''],
                           ['environment', 'mean value', '']]
        mean_values = df.groupby('environment')['place fields'].mean()
        for env, value in zip(mean_values.index, mean_values):
            table_cell_text.append([env, str(value), ''])
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

        return {'slope': main_line_slope, 'intercept': main_line_intercept}

    @staticmethod
    def make_figure(df_units, df_fields, verbose=False):

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.subplots_adjust(left=0.11, bottom=0.2, right=0.99, top=0.98)

        stat_fig, stat_axs = plt.subplots(1, 1, figsize=(8, 15))
        plt.tight_layout(pad=1.5)
        stat_axs.set_xticks([], [])
        stat_axs.set_yticks([], [])

        environment_field_density_model_parameters = FieldsDetectedAcrossEnvironments.plot_place_field_distribution(
            df_units, df_fields, ax, stat_axs
        )

        return fig, stat_fig, environment_field_density_model_parameters

    @staticmethod
    def write(fpath, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldsDetectedAcrossEnvironments'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig, environment_field_density_model_parameters = \
            FieldsDetectedAcrossEnvironments.make_figure(df_units, df_fields, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))

        return environment_field_density_model_parameters


class Remapping:

    @staticmethod
    def compute_ratemap_correlation(all_recordings, min_included_rate, min_required_bins):

        rho = []
        rho_halves = []
        shuffle_rho = []
        for recordings in all_recordings:

            ratemaps_a = []
            ratemaps_a2 = []
            ratemaps_a_1st_half = []
            ratemaps_a_2nd_half = []
            ratemap_shape = None
            for i, unit in enumerate(recordings.units):

                if recordings.first_available_recording_unit(i)['analysis']['category'] != 'place_cell':
                    continue

                if unit[0] is None:
                    ratemaps_a.append(None)
                else:
                    ratemaps_a.append(unit[0]['analysis']['spatial_ratemaps']['spike_rates_smoothed'])
                    ratemaps_a_1st_half.append(unit[0]['analysis']['spatial_ratemaps']['spike_rates_halves']['first'])
                    ratemaps_a_2nd_half.append(unit[0]['analysis']['spatial_ratemaps']['spike_rates_halves']['second'])
                    if ratemap_shape is None:
                        ratemap_shape = ratemaps_a[-1].shape

                if unit[4] is None:
                    ratemaps_a2.append(None)
                else:
                    ratemaps_a2.append(unit[4]['analysis']['spatial_ratemaps']['spike_rates_smoothed'])

            for rm_list in (ratemaps_a, ratemaps_a2):
                for i, ratemap in enumerate(rm_list):
                    if ratemap is None:
                        rm_list[i] = np.zeros(ratemap_shape, dtype=np.float64)

            for ratemap_a, ratemap_a2 in zip(ratemaps_a, ratemaps_a2):
                rho.append(spatial_correlation(ratemap_a, ratemap_a2,
                                               min_included_value=min_included_rate,
                                               min_bins=min_required_bins)[0])

            for ratemap_1st_half, ratemap_2nd_half in zip(ratemaps_a_1st_half, ratemaps_a_2nd_half):
                rho_halves.append(spatial_correlation(ratemap_1st_half, ratemap_2nd_half,
                                                      min_included_value=min_included_rate,
                                                      min_bins=min_required_bins)[0])

            shuffle(ratemaps_a2)
            for ratemap_a, ratemap_a2 in zip(ratemaps_a, ratemaps_a2):
                shuffle_rho.append(spatial_correlation(ratemap_a, ratemap_a2,
                                                       min_included_value=min_included_rate,
                                                       min_bins=min_required_bins)[0])

        df_a_halves = pd.DataFrame({'Pearson $\it{r}$': np.array(rho_halves)})
        df_a_halves['group'] = 'A 1/2 v 1/2\nintra-trial'
        df_a = pd.DataFrame({'Pearson $\it{r}$': np.array(rho)})
        df_a['group'] = "A v A'\ninter-trial"
        df_a2 = pd.DataFrame({'Pearson $\it{r}$': np.array(shuffle_rho)})
        df_a2['group'] = "A v shuffled A'"
        df = pd.concat((df_a_halves, df_a, df_a2), axis=0, ignore_index=True)
        df.dropna(inplace=True)

        return df

    @staticmethod
    def plot_ratemap_correlation(all_recordings, ax, stat_ax, min_included_rate, min_required_bins):
        """Plot shows the correlation between ratemaps:
        A 1/2 v 1/2 - same unit in environment A, ratemaps computed on first and last half of the recording.
        A v A' - same unit in environment A and environment A'
        A v shuffled A' - ratemap of one unit in environment A is correlated with a ratemap of a random unit
            from the same animal in environment A'. Only a single iteration of this permutation is performed.

        The box plot shows median, Q1, Q3 and 5-95% data range.
        """

        df = Remapping.compute_ratemap_correlation(all_recordings, min_included_rate, min_required_bins)

        groups_order = sorted(list(df['group'].unique()))

        plot_raincloud_and_stats('group', 'Pearson $\it{r}$', df, ax, stat_ax,
                                 palette=sns.color_palette(sns_other_colors[:len(groups_order)]),
                                 x_order=groups_order)

        ax.set_yticks([y for y in ax.get_yticks() if y <= 1])

        stat_ax.set_title('ratemap_correlation')

    @staticmethod
    def compute_bayes_decoding_arena_accuracy_and_peak_values(all_recordings, position_decoding_name):
        dfs, _ = BayesianPositionDecodingArenaAccuracy.compute_for_all_recordings(
            all_recordings, position_decoding_name
        )

        df_accuracy = pd.concat(dfs).reset_index()

        return df_accuracy

    @staticmethod
    def plot_bayes_decoding_arena_accuracy(df, ax, stat_ax):
        """Plots the percentage of samples decoded to each environment, separately for when animal
        was in each of the environments (except A'). Each datapoint is the percentage value for a single animal.
        """
        order = ('A', 'B', 'C', 'D')
        hue_order = ('A', 'B', 'C', 'D')

        sns.barplot(x='real environment', y='percentage', hue='decoded environment',
                    order=order, hue_order=hue_order,
                    data=df, palette=sns.color_palette(sns_environment_colors[:4]),
                    ax=ax)
        sns.stripplot(x='real environment', y='percentage', hue='decoded environment',
                      order=order, hue_order=hue_order,
                      data=df, palette=sns.color_palette(sns_environment_colors[:4]),
                      ax=ax, linewidth=2, dodge=True)
        ax.set_yscale('log')
        ax.set_ylabel('decoded samples in location (%)')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[len(hue_order):], labels[len(hue_order):],
                  ncol=len(df['decoded environment'].unique()), title='decoded environment',
                  loc='lower left', bbox_to_anchor=(0, 1))

        # Plot statistics on stat_ax

        stat_ax.set_title('bayes_decoding_arena_accuracy')
        mean_abcd_correct = np.mean(df[(df['real environment'] == df['decoded environment'])]['percentage'])
        table_cell_text = [['Mean occurance', 'environments', 'accuracy'],
                           ['correct', 'A, B, C, D', '{:.2f}'.format(mean_abcd_correct)],
                           ['', '', ''],
                           ['Environment', 'Percentage correct', ''],
                           ['A', str(np.mean(df[(df['real environment'] == df['decoded environment'])
                                                & (df['real environment'] == 'A')]['percentage'])), ''],
                           ['B', str(np.mean(df[(df['real environment'] == df['decoded environment'])
                                                & (df['real environment'] == 'B')]['percentage'])), ''],
                           ['C', str(np.mean(df[(df['real environment'] == df['decoded environment'])
                                                & (df['real environment'] == 'C')]['percentage'])), ''],
                           ['D', str(np.mean(df[(df['real environment'] == df['decoded environment'])
                                                & (df['real environment'] == 'D')]['percentage'])), '']]
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def plot_bayes_decoding_results(all_recordings, position_decoding_name, ax, stat_ax):

        df_accuracy = Remapping.compute_bayes_decoding_arena_accuracy_and_peak_values(all_recordings,
                                                                                      position_decoding_name)

        Remapping.plot_bayes_decoding_arena_accuracy(df_accuracy, ax, stat_ax)

    @staticmethod
    def make_figure(all_recordings):

        fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 2]})
        plt.subplots_adjust(left=0.07, right=0.99, bottom=0.12, top=0.83, hspace=0.4)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(12, 15))
        plt.tight_layout(pad=1.5)
        for ax in stat_axs.flatten():
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        Remapping.plot_ratemap_correlation(all_recordings, axs[0], stat_axs[0],
                                           Params.ratemap_stability_kwargs['min_included_value'],
                                           Params.ratemap_stability_kwargs['min_bins'])

        Remapping.plot_bayes_decoding_results(all_recordings, 'position_decoding_large_window',
                                              axs[1], stat_axs[1])

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, prefix='', verbose=True):

        figure_name = prefix + 'Remapping'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = Remapping.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldsPerCellAcrossEnvironmentsSimple:

    @staticmethod
    def compute_place_field_formation_propensity(df_fields, df_units, combine_environments=False,
                                                 add_silent_cells=True):

        # Create a copy of df_fields with only the relevant columns
        df = df_fields[['unit', 'animal', 'animal_unit', 'experiment_id']].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[df['category'] == 'place_cell']  # Only keep place cell fields
        df = df[['unit', 'experiment_id']]

        # Only keep fields not in exp_scales_a2
        df = df[df['experiment_id'] != 'exp_scales_a2']

        # Replace experiment_id values for plotting and rename the column
        df.replace(to_replace={'experiment_id': experiment_id_substitutes}, inplace=True)
        df.rename(columns={'experiment_id': 'environment'}, inplace=True)

        # Assign all units to a single environment if requested
        if combine_environments:
            df['environment'] = 'combined environment'

        # Compute count per unit
        df['number of fields'] = np.ones(df.shape[0])
        df = df.groupby(['unit', 'environment']).sum().reset_index()

        # Keep a separate count per unit for output
        df_count_per_unit = df.copy(deep=True)

        # Compute number of place cells with 0 fields based on how many place cells were recorded in total
        n_total_place_cells = df['unit'].unique().size
        silent_unit_environments = []
        n_silent_units_in_environment = []
        for environment, series in df.groupby('environment').count().iterrows():
            silent_unit_environments.append(environment)
            n_silent_units_in_environment.append(n_total_place_cells - series['unit'])

        # Compute count per environment and field count
        df['number of place cells'] = np.ones(df.shape[0])
        df = df.groupby(['environment', 'number of fields'])['number of place cells'].count().reset_index()

        # Add the silent cells to the count
        if add_silent_cells:
            df_place_cells = pd.concat([df, pd.DataFrame({'environment': silent_unit_environments,
                                                          'number of fields': np.zeros(len(silent_unit_environments)),
                                                          'number of place cells': n_silent_units_in_environment})],
                                       ignore_index=True)
        else:
            df_place_cells = df

        dfg = df_place_cells.groupby(['environment']).sum().reset_index()[['environment', 'number of place cells']]
        dfg.rename(columns={'number of place cells': 'total place cells'}, inplace=True)
        df_place_cells = df_place_cells.merge(dfg, how='left', on='environment')
        df_place_cells['proportion of place cells'] = \
            df_place_cells['number of place cells'] / df_place_cells['total place cells']

        return df_place_cells, df_count_per_unit

    @staticmethod
    def plot(df_units, df_fields, ax, stat_ax):
        df_by_environment, df_count_per_unit = \
            FieldsPerCellAcrossEnvironmentsSimple.compute_place_field_formation_propensity(df_fields, df_units,
                                                                                           add_silent_cells=False)
        df_combined, df_count_per_unit_combined = \
            FieldsPerCellAcrossEnvironmentsSimple.compute_place_field_formation_propensity(
                df_fields, df_units, combine_environments=True, add_silent_cells=False
            )
        df = pd.concat([df_by_environment, df_combined], 0, ignore_index=True, sort=True)
        df_count_per_unit = pd.concat([df_count_per_unit, df_count_per_unit_combined], 0, ignore_index=True, sort=True)

        environments = df['environment'].unique()
        colors = sns_environment_colors[:len(environments)]
        colors_dict = {key: color for key, color in zip(environments, colors)}

        environment_mean_field_counts = {}
        environment_proportion_multifield_cells = {}
        for environment in environments:
            idx = df_count_per_unit['environment'] == environment
            environment_mean_field_counts[environment] = df_count_per_unit.loc[idx, 'number of fields'].mean()
            environment_proportion_multifield_cells[environment] = \
                np.sum(df_count_per_unit.loc[idx, 'number of fields'] > 1) / np.sum(idx)

        second_to_last_field_count = 11
        last_field_count = 13
        last_field_count_label = None
        last_field_count_value = None

        for i, environment in enumerate(environments):
            df_tmp = df.loc[df['environment'] == environment].copy()
            df_tmp = df_tmp.sort_values('number of fields')

            # Crop extra values
            if np.any(df_tmp['number of fields'] > second_to_last_field_count):
                if last_field_count_label is not None:
                    raise Exception('last_value_label should only be assgined once, for combined environment')

                last_value_label = np.max(df_tmp['number of fields'])
                last_field_count_value = df_tmp['proportion of place cells'].values[-1]

                df_tmp = df_tmp.loc[df_tmp['number of fields'] <= second_to_last_field_count].copy()
                df_tmp_row = df_tmp.iloc[0:1].copy(deep=True)
                df_tmp_row.loc[df_tmp_row.index[0], 'number of fields'] = last_field_count
                df_tmp_row.loc[df_tmp_row.index[0], 'proportion of place cells'] = last_field_count_value

            ax.plot(df_tmp['number of fields'], df_tmp['proportion of place cells'], color=colors_dict[environment],
                    label=environment, marker='o', linewidth=2, zorder=-i)

        if last_field_count_value is None or last_value_label is None:
            raise Exception('last_field_count_value was never set. Must include combined environment.')

        ax.legend(loc='upper right')

        ax.set_xlabel('number of fields')
        ax.set_ylabel('proportion of active place cells')
        ax.set_xticks(np.arange(1, last_field_count, 2))
        xtick_labels = np.arange(1, last_field_count, 2)
        xtick_labels[-1] = last_value_label
        ax.set_xticklabels(xtick_labels)

        table_cell_text = [['Environment', 'Mean field count']]
        for environment, mean_count in environment_mean_field_counts.items():
            table_cell_text.append([environment, '{:.3f}'.format(mean_count)])

        table_cell_text.append(['', ''])
        table_cell_text.append(['Environment', 'proportion multi-field'])
        for environment, proportion_multifield in environment_proportion_multifield_cells.items():
            table_cell_text.append([environment, '{:.3f}'.format(proportion_multifield)])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def make_figure(df_units, df_fields):

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        plt.subplots_adjust(left=0.13, bottom=0.15, right=0.99, top=0.90, wspace=0.3)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FieldsPerCellAcrossEnvironmentsSimple.plot(df_units, df_fields, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldsPerCellAcrossEnvironmentsSimple'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldsPerCellAcrossEnvironmentsSimple.make_figure(df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class ConservationOfFieldFormationPropensity(object):

    @staticmethod
    def get_dataframe(df_units, df_fields, environment_field_density_model_parameters):

        # Create a copy of df_fields with only the relevant columns
        df = df_fields[['animal', 'animal_unit', 'experiment_id']].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        # Only keep place cell fields
        df = df.loc[df['category'] == 'place_cell', ['animal', 'animal_unit', 'experiment_id']]

        # Drop fields in smallest environment
        df = df[(df['experiment_id'] != 'exp_scales_a') & (df['experiment_id'] != 'exp_scales_a2')]

        # Count fields per unit in each environment
        df['count'] = 1
        df = df.groupby(['animal', 'animal_unit', 'experiment_id']).sum().reset_index()

        # Compute field formation propensity
        environment_areas_corrected = \
            FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
                parameters=environment_field_density_model_parameters
            )
        new_environment_areas_corrected = {}
        for environment, area in environment_areas_corrected.items():
            if environment in experiment_id_substitutes_inverse:
                new_environment_areas_corrected[experiment_id_substitutes_inverse[environment]] = area
        df['environment_areas'] = df['experiment_id'].map(new_environment_areas_corrected)
        df['field_formation_propensity'] = df['count'] / df['environment_areas']

        return df

    @staticmethod
    def plot(df_units, df_fields, environment_field_density_model_parameters, ax, stat_ax, n_shuffles=1000):

        df = ConservationOfFieldFormationPropensity.get_dataframe(df_units, df_fields,
                                                                  environment_field_density_model_parameters)
        measure = 'field_formation_propensity'

        required_count = df['experiment_id'].unique().size
        df = df[df.groupby(['animal', 'animal_unit'])[measure].transform('size') == required_count].reset_index(
            drop=True)

        df_real = df.groupby(['animal', 'animal_unit']).std().reset_index().copy(deep=True)

        # Create multiple shuffled unit variance results
        shuffle_indices = {animal: {} for animal in df['animal'].unique()}
        for animal in df['animal'].unique():
            for experiment_id in df.loc[df['animal'] == animal, 'experiment_id'].unique():
                shuffle_indices[animal][experiment_id] = \
                    np.where((df['animal'] == animal) & (df['experiment_id'] == experiment_id))[0]

        df_shuffle = []
        for _ in tqdm(range(n_shuffles)):

            for animal in df['animal'].unique():
                for experiment_id in shuffle_indices[animal].keys():
                    previous_indices = shuffle_indices[animal][experiment_id]
                    new_indices = shuffle_indices[animal][experiment_id].copy()
                    np.random.shuffle(new_indices)
                    df.loc[previous_indices, 'animal_unit'] = df.loc[new_indices, 'animal_unit'].values

            df_shuffle.append(df.groupby(['animal', 'animal_unit']).std().reset_index())

        df_shuffle = pd.concat(df_shuffle, axis=0, ignore_index=True)

        # Compute Mann-Whitney rank test
        statistic, pvalue = mannwhitneyu(df_real[measure], df_shuffle[measure], alternative='less')

        # Plot results

        df_real['group'] = 'data'
        df_shuffle['group'] = 'shuffle'
        df = pd.concat([df_real, df_shuffle], axis=0, ignore_index=True)

        sns.histplot(
            data=df, x=measure, hue='group', element="step", fill=False,
            cumulative=True, stat='density', common_norm=False, ax=ax
        )

        # plot_normalised_histograms_of_real_and_shuffle_data(df_real[measure], df_shuffle[measure], ax, bins=10)
        ax.set_ylabel('cumulative proportion of cells')
        ax.set_xlabel('st.dev.(place fields / m$^2$)')
        # ax.legend()

        # Plot stats

        stat_ax.set_title(measure)
        table_cell_text = [
            ['Mann-Whitney test', ''],
            ['U value', '{:.2e}'.format(statistic)],
            ['p-value', '{:.2e}'.format(pvalue)],
            ['', ''],
            ['Total samples', str(df_real.shape[0])],
            ['n_shuffles', str(n_shuffles)]
        ]

        for animal in sorted(df_real['animal'].unique()):
            table_cell_text += [[animal, str(np.sum(df_real['animal'] == animal))]]

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def make_figure(df_units, df_fields, environment_field_density_model_parameters, verbose=False):

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0.32, bottom=0.2, right=0.98, top=0.9)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        ConservationOfFieldFormationPropensity.plot(df_units, df_fields, environment_field_density_model_parameters,
                                                    ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, environment_field_density_model_parameters, prefix='', verbose=True):

        figure_name = prefix + 'ConservationOfFieldFormationPropensity'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = ConservationOfFieldFormationPropensity.make_figure(
            df_units, df_fields, environment_field_density_model_parameters, verbose=verbose
        )
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldsPerCellAcrossEnvironments:

    @staticmethod
    def compute_distribution_from_field_counts(field_counts):
        field_counts, unit_counts = np.unique(field_counts, return_counts=True)
        unit_count_by_field_count = np.zeros(np.max(field_counts) + 1)
        unit_count_by_field_count[field_counts] = unit_counts
        field_count_distribution = unit_count_by_field_count / np.sum(unit_count_by_field_count)

        return field_count_distribution

    @staticmethod
    def predict_probability_of_field_count(areas, counts, gamma_shape, gamma_scale):
        """Returns directly the Poisson(Gamma) probability mass function value at specific counts.

        Rich et al 2014 DOI: 10.1126/science.1255635
        Supplementary materials page 8 equation with slight correction: (1 - p) instead of (p - 1).
        """
        gamma_rates = 1 / (gamma_scale * areas)
        r = gamma_shape
        p_values = gamma_rates / (gamma_rates + 1)
        probabilities = [
            (gamma_function(r + x) / (gamma_function(r) * gamma_function(x + 1)))
            * np.power(p, r) * np.power(1 - p, x)
            for x, p in zip(counts, p_values)
        ]

        return np.array(probabilities)

    @staticmethod
    def predict_field_count_distribution_with_gamma_poisson(area, gamma_shape, gamma_scale, max_counts=100):
        """Returns an array of field count distribution among units
        """
        return FieldsPerCellAcrossEnvironments.predict_probability_of_field_count(
            area * np.ones(max_counts), np.arange(max_counts), gamma_shape, gamma_scale
        )

    @staticmethod
    def predict_field_count_distribution_with_equal_poisson(area, poisson_rate, max_counts=100):
        """Returns an array of field count distribution among units
        """
        return poisson.pmf(np.arange(max_counts), poisson_rate * area)

    @staticmethod
    def compute_log_likelihood_of_data_given_gamma_parameters(areas, field_counts, gamma_shape, gamma_scale):
        """
        Returns the sum over natural logarithms of likelihood estimations with given gamma parameters.

        :param areas: shape (N,) of environment sizes where N cells were observed to be active
        :param field_counts: shape (N,) of field counts of N cells.
        :param gamma_shape:
        :param gamma_scale:
        :return: log_likelihood
        """
        probabilities = FieldsPerCellAcrossEnvironments.predict_probability_of_field_count(
            areas, field_counts, gamma_shape, gamma_scale
        )
        return np.sum(np.log(probabilities))

    @staticmethod
    def compute_log_likelihood_of_data_given_poisson_rate(areas, field_counts, poisson_rate):
        """
        Returns the sum over natural logarithms of likelihood estimations with given poisson rate parameter.

        :param areas:
        :param field_counts:
        :param poisson_rate: Poisson rate per unit area that is constant for all cells.
        :return: log_likelihood
        """
        probabilities = poisson.pmf(field_counts, poisson_rate * areas)
        return np.sum(np.log(probabilities))

    @staticmethod
    def construct_negative_log_likelihood_model_fitting_method(model_name):
        if model_name == 'gamma-poisson':
            return lambda x, areas, field_counts: \
                -FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_gamma_parameters(
                    areas, field_counts, x[0], x[1]
                )
        elif model_name == 'equal-poisson':
            return lambda x, areas, field_counts: \
                -FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_poisson_rate(
                    areas, field_counts, x[0]
                )
        else:
            raise ValueError('Expected model name gamma-poisson or equal-poisson but got {}'.format(model_name))

    @staticmethod
    def predict_field_count_distributions_in_multiple_areas(areas, model_name, params, total_units=100000):
        """Returns an array of field count of same units in multiple areas
        """
        if model_name == 'gamma-poisson':
            propensities = gamma.rvs(a=params[0], loc=0, scale=params[1], size=total_units)
        elif model_name == 'equal-poisson':
            propensities = params[0] * np.ones(total_units)
        else:
            raise ValueError('Expected model name gamma-poisson or equal-poisson but got {}'.format(model_name))
        field_counts_all = [poisson.rvs(propensities * area) for area in areas]

        # Compute field count distribution in each area separately using all units
        field_count_distributions_per_area_all = []
        for field_counts_in_area in field_counts_all:
            field_count_distributions_per_area_all.append(
                FieldsPerCellAcrossEnvironments.compute_distribution_from_field_counts(field_counts_in_area)
            )

        # Compute field count distribution in each area separately using only the units with one field
        # in at least one of the areas

        idx_active = np.zeros(total_units, dtype=np.bool)
        for field_counts_in_area_all in field_counts_all:
            idx_active = np.logical_or(idx_active, field_counts_in_area_all > 0)

        field_count_distributions_per_area_active = []
        for field_counts_in_area in field_counts_all:
            field_count_distributions_per_area_active.append(
                FieldsPerCellAcrossEnvironments.compute_distribution_from_field_counts(
                    field_counts_in_area[idx_active]
                )
            )

        return field_count_distributions_per_area_all, field_count_distributions_per_area_active

    @staticmethod
    def predict_field_count_distributions_in_multiple_areas_for_active_units(
            areas, gamma_shape, gamma_scale, environment_field_density_model_parameters, total_units=100000,
    ):
        """Returns an array of field count of same units in multiple areas, only using modelled
        units that had at least one field in one of the experiment areas
        """

        environment_areas_corrected = \
            FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
                parameters=environment_field_density_model_parameters
            )
        environment_areas_corrected = [
            environment_areas_corrected[environment] for environment in ('A', 'B', 'C', 'D')
        ]

        propensities = gamma.rvs(a=gamma_shape, loc=0, scale=gamma_scale, size=total_units)
        field_counts_all = [poisson.rvs(propensities * area) for area in environment_areas_corrected]

        # Compute field count distribution in each area separately using all units
        field_count_distributions_per_area_all = []
        for field_counts_in_area in field_counts_all:
            field_count_distributions_per_area_all.append(
                FieldsPerCellAcrossEnvironments.compute_distribution_from_field_counts(field_counts_in_area)
            )

        # Compute field count distribution in each area separately using only the units with one field
        # in at least one of the areas

        idx_active = np.zeros(total_units, dtype=np.bool)
        for field_counts_in_area_all in field_counts_all:
            idx_active = np.logical_or(idx_active, field_counts_in_area_all > 0)

        propensities = propensities[idx_active]
        field_counts_all = [poisson.rvs(propensities * area) for area in areas]

        field_count_distributions_per_area_active = []
        for field_counts_in_area in field_counts_all:
            field_count_distributions_per_area_active.append(
                FieldsPerCellAcrossEnvironments.compute_distribution_from_field_counts(
                    field_counts_in_area
                )
            )

        return field_count_distributions_per_area_all, field_count_distributions_per_area_active

    @staticmethod
    def plot_gamma_pdf(gamma_shape, gamma_scale, ax):

        x = np.linspace(0, gamma_shape * gamma_scale * 5, 1000)
        y = gamma.pdf(x, gamma_shape, loc=0, scale=gamma_scale)

        ax.plot(x, y)

        ax.set_xlim((0, x.max()))
        ax.set_ylim((0, ax.get_ylim()[1]))

        ax.set_ylabel('Gamma pdf')
        ax.set_xlabel('place fields / m$^2$')

        ax.text(0.9, 0.9, 'shape = {:.3f}\nscale = {:.3f}'.format(gamma_shape, gamma_scale),
                ha='right', va='top', transform=ax.transAxes)

    @staticmethod
    def plot_field_formation_propensities(df_fields, df_units, environment_field_density_model_parameters, ax, stat_ax):

        df_by_environment, _ = \
            FieldsPerCellAcrossEnvironmentsSimple.compute_place_field_formation_propensity(df_fields, df_units)
        df_combined, df_count_per_unit = \
            FieldsPerCellAcrossEnvironmentsSimple.compute_place_field_formation_propensity(
                df_fields, df_units, combine_environments=True
            )
        df = pd.concat([df_by_environment, df_combined], 0, ignore_index=True, sort=True)

        areas_corrected = FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
            parameters=environment_field_density_model_parameters
        )
        areas_not_corrected = FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
            parameters=None
        )
        df_count_per_unit['areas_corrected'] = df_count_per_unit['environment'].map(areas_corrected)
        df_count_per_unit['areas_not_corrected'] = df_count_per_unit['environment'].map(areas_not_corrected)

        # Drop cell counts with 0 fields in each environment and recompute proportion of cells
        dfs = []
        areas_corrected_list = []
        areas_not_corrected_list = []
        field_count_distribution_prediction_list = []
        for environment in list(df['environment'].unique()):
            idx = df['environment'] == environment

            field_counts = np.int16(df.loc[idx, 'number of fields'])
            unit_percentages = df.loc[idx, 'proportion of place cells']
            field_count_distribution = np.zeros(np.max(field_counts) + 1)
            field_count_distribution[field_counts] = unit_percentages.values
            field_count_distribution = field_count_distribution[1:]
            field_count_distribution = field_count_distribution / np.sum(field_count_distribution)

            df_tmp = pd.DataFrame({'number of fields': np.arange(field_count_distribution.size) + 1,
                                   'proportion of active place cells': field_count_distribution})
            df_tmp['environment'] = environment

            areas_corrected_list.append(areas_corrected[environment])
            areas_not_corrected_list.append(areas_not_corrected[environment])
            field_count_distribution_prediction_list.append(field_count_distribution)

            dfs.append(df_tmp)

        df = pd.concat(dfs, 0, ignore_index=True, sort=True)
        df['values'] = 'data - place cells in environment'

        environments = sorted(df['environment'].unique())
        environments_real = sorted(list(set(environments) - {'combined environment'}))

        # Fit model parameters using maximum likelihood estimation

        print('Fitting Poisson rate with field density correction')
        res = minimize(
            FieldsPerCellAcrossEnvironments.construct_negative_log_likelihood_model_fitting_method('equal-poisson'),
            np.array([0.1]),
            args=(df_count_per_unit['areas_corrected'].values, df_count_per_unit['number of fields'].values),
            bounds=((np.finfo(np.float32).resolution, None),),
            options={'disp': True}
        )

        poisson_rate, = res.x

        print('Fitting Poisson rate without field density correction')
        res = minimize(
            FieldsPerCellAcrossEnvironments.construct_negative_log_likelihood_model_fitting_method('equal-poisson'),
            np.array([0.1]),
            args=(df_count_per_unit['areas_not_corrected'].values, df_count_per_unit['number of fields'].values),
            bounds=((np.finfo(np.float32).resolution, None),),
            options={'disp': True}
        )

        not_corrected_poisson_rate, = res.x

        print('Fitting Gamma parameters with field density correction')
        res = minimize(
            FieldsPerCellAcrossEnvironments.construct_negative_log_likelihood_model_fitting_method('gamma-poisson'),
            np.array([1, 0.1]),
            args=(df_count_per_unit['areas_corrected'].values, df_count_per_unit['number of fields'].values),
            bounds=((np.finfo(np.float32).resolution, None), (np.finfo(np.float32).resolution, None)),
            options={'disp': True}
        )

        gamma_shape, gamma_scale = res.x

        print('Fitting Gamma parameters without field density correction')
        res = minimize(
            FieldsPerCellAcrossEnvironments.construct_negative_log_likelihood_model_fitting_method('gamma-poisson'),
            np.array([1, 0.1]),
            args=(df_count_per_unit['areas_not_corrected'].values, df_count_per_unit['number of fields'].values),
            bounds=((np.finfo(np.float32).resolution, None), (np.finfo(np.float32).resolution, None)),
            options={'disp': True}
        )

        not_corrected_gamma_shape, not_corrected_gamma_scale = res.x

        # Make predictions
        for model_name in ('equal-Poisson place cells in environment', 'gamma-Poisson place cells in environment'):

            dfs = []
            for environment in environments:

                if model_name == 'equal-Poisson place cells in environment':
                    field_count_distribution_full = \
                        FieldsPerCellAcrossEnvironments.predict_field_count_distribution_with_equal_poisson(
                            areas_corrected[environment], poisson_rate
                        )
                elif model_name == 'gamma-Poisson place cells in environment':
                    field_count_distribution_full = \
                        FieldsPerCellAcrossEnvironments.predict_field_count_distribution_with_gamma_poisson(
                            areas_corrected[environment], gamma_shape, gamma_scale
                        )
                else:
                    raise ValueError()

                # Compute the distribution normalised to the total number of units with at least one field
                sum_without_silent_cells = 1 - field_count_distribution_full[0]
                field_count_distribution_less = field_count_distribution_full[1:]
                field_count_distribution_less = field_count_distribution_less / sum_without_silent_cells
                field_count_distribution_less = np.concatenate([[np.nan], field_count_distribution_less])

                df_tmp = pd.DataFrame({'number of fields': np.arange(field_count_distribution_full.size),
                                       'proportion of all place cells': field_count_distribution_full,
                                       'proportion of active place cells': field_count_distribution_less})
                df_tmp['environment'] = environment

                dfs.append(df_tmp)

            df_tmp = pd.concat(dfs, 0, ignore_index=True, sort=True)
            df_tmp['values'] = model_name

            df = pd.concat([df, df_tmp], 0, ignore_index=True, sort=True)

        # Plot
        fig = ax.figure
        ax.axis('off')
        width_ratios = [df.loc[(df['environment'] == environment)
                               & (df['values'] == 'data - place cells in environment'), 'number of fields'].max()
                        for environment in environments_real]

        gs = GridSpecFromSubplotSpec(2, 1, ax, height_ratios=[1, 3], hspace=0.4)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1])
        ax_top.axis('off')
        ax_bottom.axis('off')

        gs = GridSpecFromSubplotSpec(1, len(width_ratios), ax_bottom, wspace=0.08, width_ratios=width_ratios)
        axs_real_environments = [fig.add_subplot(g) for g in gs]

        gs = GridSpecFromSubplotSpec(1, 2, ax_top, wspace=0.2, width_ratios=[3, 1])
        axs_top_left = fig.add_subplot(gs[0])
        axs_top_right = fig.add_subplot(gs[1])

        colors = sns_environment_colors[:len(environments)]
        colors_dict = {key: color for key, color in zip(environments, colors)}

        for ax, environment in zip(axs_real_environments + [axs_top_left], environments):
            idx = df['environment'] == environment
            idx = idx & (df['number of fields'] > 0)

            label = 'data - place cells in environment'
            n_fields = df.loc[idx & (df['values'] == label), 'number of fields'].values
            prop_units = df.loc[idx & (df['values'] == label), 'proportion of active place cells'].values
            ax.bar(n_fields, prop_units, align='center', width=np.ones(n_fields.size) * 0.9,
                   color=colors_dict[environment])

            label = 'equal-Poisson place cells in environment'
            n_fields = df.loc[idx & (df['values'] == label), 'number of fields'].values
            prop_units = df.loc[idx & (df['values'] == label), 'proportion of active place cells'].values
            ax.plot(n_fields, prop_units, 'r', linewidth=2, label='equal-Poisson')

            label = 'gamma-Poisson place cells in environment'
            n_fields = df.loc[idx & (df['values'] == label), 'number of fields'].values
            prop_units = df.loc[idx & (df['values'] == label), 'proportion of active place cells'].values
            ax.plot(n_fields, prop_units, 'k', linewidth=2, label='gamma-Poisson')

            ax.set_xlim((0.5, df.loc[idx & (df['values'] == 'data - place cells in environment'),
                                     'number of fields'].max() + 0.5))
            ax.set_xticks(np.arange(1, df.loc[idx & (df['values'] == 'data - place cells in environment'),
                                              'number of fields'].max(),
                                    2))

            ax.set_title('( {} )'.format(environment))

        axs_top_left.legend(loc='upper right', framealpha=1)
        axs_real_environments[3].legend(loc='upper right', framealpha=1)

        axs_real_environments[0].set_ylabel('proportion of cells')
        axs_real_environments[2].set_xlabel('number of fields')
        axs_top_left.set_ylabel('proportion of cells')
        axs_top_left.set_xlabel('number of fields')

        ylim = get_max_ylim(axs_real_environments)
        for ax in axs_real_environments:
            ax.set_ylim(ylim)

        for ax in axs_real_environments[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)

        # Plot Gamma distribution pdf to inset
        FieldsPerCellAcrossEnvironments.plot_gamma_pdf(gamma_shape, gamma_scale, axs_top_right)

        # Compute statistics for all models

        # Maximum loglikelihood
        poisson_maximum_loglikelihood = \
            FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_poisson_rate(
                df_count_per_unit['areas_corrected'].values, df_count_per_unit['number of fields'].values,
                poisson_rate
            )
        not_corrected_poisson_maximum_loglikelihood = \
            FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_poisson_rate(
                df_count_per_unit['areas_not_corrected'].values, df_count_per_unit['number of fields'].values,
                not_corrected_poisson_rate
            )
        gamma_maximum_loglikelihood = \
            FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_gamma_parameters(
                df_count_per_unit['areas_corrected'].values, df_count_per_unit['number of fields'].values,
                gamma_shape, gamma_scale
            )
        not_corrected_gamma_maximum_loglikelihood = \
            FieldsPerCellAcrossEnvironments.compute_log_likelihood_of_data_given_gamma_parameters(
                df_count_per_unit['areas_not_corrected'].values, df_count_per_unit['number of fields'].values,
                not_corrected_gamma_shape, not_corrected_gamma_scale
            )

        model_names = ('poisson', 'not_corrected_poisson', 'gamma', 'not_corrected_gamma')
        model_parameter_count = (1, 1, 2, 2)
        model_loglikelihoods = (
            poisson_maximum_loglikelihood,
            not_corrected_poisson_maximum_loglikelihood,
            gamma_maximum_loglikelihood,
            not_corrected_gamma_maximum_loglikelihood
        )

        # AIC
        aic = {}
        for name, loglikelihood, parameter_count in zip(model_names, model_loglikelihoods, model_parameter_count):
            aic[name] = 2 * parameter_count - 2 * loglikelihood

        # BIC
        bic = {}
        for name, loglikelihood, parameter_count in zip(model_names, model_loglikelihoods, model_parameter_count):
            bic[name] = parameter_count * np.log(df_count_per_unit.shape[0]) - 2 * loglikelihood

        # Plot statistics for all models
        table_cell_text = [['equal-Poisson', 'corrected'],
                           ['poisson_rate', str(poisson_rate)],
                           ['poisson_maximum_loglikelihood', str(poisson_maximum_loglikelihood)],
                           ['akaike information criterion', str(aic['poisson'])],
                           ['bayesian information criterion', str(bic['poisson'])],
                           ['', ''],
                           ['equal-Poisson', 'not corrected'],
                           ['not_corrected_poisson_rate', str(not_corrected_poisson_rate)],
                           ['not_corrected_poisson_maximum_loglikelihood',
                            str(not_corrected_poisson_maximum_loglikelihood)],
                           ['akaike information criterion', str(aic['not_corrected_poisson'])],
                           ['bayesian information criterion', str(bic['not_corrected_poisson'])],
                           ['', ''],
                           ['gamma-Poisson', 'corrected'],
                           ['gamma_shape', str(gamma_shape)],
                           ['gamma_scale', str(gamma_scale)],
                           ['gamma_maximum_loglikelihood', str(gamma_maximum_loglikelihood)],
                           ['akaike information criterion', str(aic['gamma'])],
                           ['bayesian information criterion', str(bic['gamma'])],
                           ['', ''],
                           ['gamma-Poisson', 'not corrected'],
                           ['not_corrected_gamma_shape', str(not_corrected_gamma_shape)],
                           ['not_corrected_gamma_scale', str(not_corrected_gamma_scale)],
                           ['not_corrected_gamma_maximum_loglikelihood',
                            str(not_corrected_gamma_maximum_loglikelihood)],
                           ['akaike information criterion', str(aic['not_corrected_gamma'])],
                           ['bayesian information criterion', str(bic['not_corrected_gamma'])],
                           ['', ''],
                           ['N cells to fit models', str(df_count_per_unit.shape[0])]]
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

        gamma_model_fit = {
            'gamma_shape': gamma_shape, 'gamma_scale': gamma_scale,
            'not_corrected_gamma_shape': not_corrected_gamma_shape,
            'not_corrected_gamma_scale': not_corrected_gamma_scale,
            'poisson_rate': poisson_rate,
            'not_corrected_poisson_rate': not_corrected_poisson_rate
        }

        return gamma_model_fit

    @staticmethod
    def make_figure(df_units, df_fields, environment_field_density_model_parameters, verbose=False):

        fig, ax = plt.subplots(1, 1, figsize=(9, 7))
        plt.subplots_adjust(left=0.08, bottom=0.08, right=0.99, top=0.95)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        gamma_model_fit = \
            FieldsPerCellAcrossEnvironments.plot_field_formation_propensities(
                df_fields, df_units, environment_field_density_model_parameters, ax, stat_ax
            )

        return fig, stat_fig, gamma_model_fit

    @staticmethod
    def write(fpath, df_units, df_fields, environment_field_density_model_parameters, prefix='', verbose=True):

        figure_name = prefix + 'FieldsPerCellAcrossEnvironments'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig, gamma_model_fit = FieldsPerCellAcrossEnvironments.make_figure(
            df_units, df_fields, environment_field_density_model_parameters, verbose=verbose
        )
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))

        return gamma_model_fit


class PlaceCellsDetectedAcrossEnvironments(object):

    @staticmethod
    def plot_place_cell_recruitment(df_units, df_fields, environment_field_density_model_parameters,
                                    gamma_model_fit, ax, stat_ax):
        """Plots the percentage of place cells detected in each animal that have a field in each environment.

        The inset shows the projection from this model to 100% recruitment.
        Units of inset axes are the same as main axes.
        """

        df, _ = \
            FieldsDetectedAcrossEnvironments.get_field_or_cell_count_per_environment_and_animal(
                df_units, df_fields
            )
        df = df.copy(deep=True)

        # Compute prediction proportion of units active in each environment for plotting

        environment_areas = FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction()
        plot_areas = np.array([environment_areas[environment] for environment in ('A', 'B', 'C', 'D')])
        # TODO: Uncomment below to compute for environments of all sizes up to 9 m2.
        # plot_areas = np.linspace(0, 9, 1000)
        areas_corrected = np.array([
            FieldsDetectedAcrossEnvironments.compute_environment_area_with_field_density_correction(
                physical_area, environment_field_density_model_parameters
            )
            for physical_area in plot_areas
        ])

        _, field_count_distributions_per_area_active = \
            FieldsPerCellAcrossEnvironments.predict_field_count_distributions_in_multiple_areas(
                areas_corrected, 'gamma-poisson', (gamma_model_fit['gamma_shape'], gamma_model_fit['gamma_scale'])
            )
        eval_recruitment = np.array(
            [(1 - field_count_distribution[0])
             for field_count_distribution in field_count_distributions_per_area_active]
        )

        environment_areas_corrected = \
            FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
                parameters=environment_field_density_model_parameters
            )
        df['corrected_environment_size'] = \
            np.array([environment_areas_corrected[environment] for environment in df['environment']])
        environment_areas_not_corrected = \
            FieldsDetectedAcrossEnvironments.compute_environment_areas_with_field_density_correction(
                parameters=None
            )
        df['not_corrected_environment_size'] = \
            np.array([environment_areas_not_corrected[environment] for environment in df['environment']])

        # Compute prediction proportion of units active in each environment for each sample to compute MSE
        _, field_count_distributions_per_area_active = \
            FieldsPerCellAcrossEnvironments.predict_field_count_distributions_in_multiple_areas(
                df['corrected_environment_size'], 'gamma-poisson',
                (gamma_model_fit['gamma_shape'], gamma_model_fit['gamma_scale'])
            )
        df['model_prediction'] = np.array(
            [(1 - field_count_distribution[0])
             for field_count_distribution in field_count_distributions_per_area_active]
        )

        # Compute predictions for each sample also without field density correction
        _, not_corrected_field_count_distributions_per_area_active = \
            FieldsPerCellAcrossEnvironments.predict_field_count_distributions_in_multiple_areas(
                df['not_corrected_environment_size'], 'gamma-poisson',
                (gamma_model_fit['not_corrected_gamma_shape'], gamma_model_fit['not_corrected_gamma_scale'])
            )
        df['not_corrected_model_prediction'] = np.array(
            [(1 - field_count_distribution[0])
             for field_count_distribution in not_corrected_field_count_distributions_per_area_active]
        )

        # Compute mean squared error for recruitment curve

        model_residuals = np.abs(df['model_prediction'] - df['place cells'])
        model_mean_squared_error = np.mean(model_residuals ** 2)

        not_corrected_model_residuals = np.abs(df['not_corrected_model_prediction'] - df['place cells'])
        not_corrected_model_mean_squared_error = np.mean(not_corrected_model_residuals ** 2)

        # Compute full recruitment curve

        eval_full_recruitment = []
        eval_full_recruitment_area = np.arange(0.1, 10.01, 0.1) ** 2
        for area in eval_full_recruitment_area:
            eval_full_recruitment.append(
                1 - FieldsPerCellAcrossEnvironments.predict_field_count_distribution_with_gamma_poisson(
                    area, gamma_model_fit['gamma_shape'], gamma_model_fit['gamma_scale'], max_counts=1
                )[0]
            )

        eval_full_recruitment = np.array(eval_full_recruitment)

        recruitment_99 = np.min(eval_full_recruitment_area[eval_full_recruitment > 0.99])

        # Plot data

        environments = sorted(np.unique(df['environment']))
        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}
        ax.scatter(df['environment_size'] + np.random.uniform(-0.2, 0.2, df['environment_size'].size),
                   df['place cells'], s=50, c=[colors_dict[x] for x in df['animal']],
                   linewidth=1, edgecolors='black', zorder=1)

        # Plot model fits

        model_points = ax.scatter(plot_areas, eval_recruitment, marker='X', color='black',
                                  label='gamma-Poisson', s=150, zorder=-1)

        ax.legend(handles=[model_points], loc='upper left')

        # Compute recruitment estimation and plot to an inset axes
        ax_inset = inset_axes(ax, width='40%', height='40%', loc='lower right')
        ax_inset.plot(eval_full_recruitment_area, eval_full_recruitment, color='black')
        ax_inset.set_ylabel('proportion of\nall place cells')
        ax_inset.set_xlabel('area (m$^2$)')
        ax_inset.xaxis.set_label_position('top')
        ax_inset.xaxis.set_ticks_position('top')
        ax_inset.set_ylim((0, 1))
        ax_inset.set_xscale('log')
        ax_inset.set_xlim((eval_full_recruitment_area[0], eval_full_recruitment_area[-1]))
        xtick_values = [0.1, 1, 10, 100]
        ax_inset.set_xticks(xtick_values)

        # Adjust axes parameters

        ax.set_xlim((0, ax.get_xlim()[1]))
        ax.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax.set_xticklabels(experiment_ids_with_areas_ticks['ticklabels'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('environment, size (m$^2$)')
        ax.set_ylabel('proportion of active place cells')
        ax.set_ylim((0, 1))

        plt.setp(ax_inset.get_xticklabels(), ha='left')

        # Compute stats

        kruskal_h_value, kruskal_pvalue = \
            kruskal(*[df[df['environment'] == group]['place cells'] for group in environments])

        df_sorted = df.sort_values('animal')
        friedman_chisq_value, friedman_pvalue = \
            friedmanchisquare(*[df_sorted[df_sorted['environment'] == group]['place cells']
                                for group in environments])

        # Plot stats to stat_ax

        stat_ax.set_title('Place cell recruitment')
        table_cell_text = [['Kruskal-Wallis test', 'H-value', 'p-value'],
                           ['', '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
                           ['', '', ''],
                           ['Friedman test test', 'chi-square statistic', 'p-value'],
                           ['', '{:.2e}'.format(friedman_chisq_value), '{:.2e}'.format(friedman_pvalue)],
                           ['', '', ''],
                           ['', 'model_mean_squared_error', '{:.3e}'.format(model_mean_squared_error)],
                           ['', 'not_corrected_model_mean_squared_error',
                            '{:.3e}'.format(not_corrected_model_mean_squared_error)],
                           ['', '', ''],
                           ['99% recruitment area size',
                            '{:.1f} m$^2$'.format(np.max(recruitment_99)), '']]
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def make_figure(df_units, df_fields, environment_field_density_model_parameters, gamma_model_fit, verbose=False):

        fig, ax = plt.subplots(1, 1, figsize=(6.2, 5))
        plt.subplots_adjust(left=0.11, bottom=0.21, right=0.96, top=0.98)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(12, 6))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        PlaceCellsDetectedAcrossEnvironments.plot_place_cell_recruitment(
            df_units, df_fields, environment_field_density_model_parameters,
            gamma_model_fit, ax, stat_ax
        )

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, environment_field_density_model_parameters,
              gamma_model_fit, prefix='', verbose=True):

        figure_name = prefix + 'PlaceCellsDetectedAcrossEnvironments'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = PlaceCellsDetectedAcrossEnvironments.make_figure(
            df_units, df_fields, environment_field_density_model_parameters, gamma_model_fit, verbose=verbose
        )
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldDistributionWithinEnvironments(object):

    @staticmethod
    def compute(all_recordings, df_units, df_fields, bin_size, combined=False):
        df = PlaceFieldPeakDistribution.compute(all_recordings, df_fields, df_units,
                                                'centroids', bin_size=bin_size, combined=combined)
        del df['field peak proportion of environment per m^2']
        df.rename(columns={'field peak proportion of total per m^2': 'density'}, inplace=True)

        compute_distances_to_landmarks(df, np.stack((df['x_coord'].values, df['y_coord'].values), axis=1))

        return df

    @staticmethod
    def plot(all_recordings, df_units, df_fields, fig, ax, stat_axs):

        df = FieldDistributionWithinEnvironments.compute(all_recordings, df_units, df_fields, 4,
                                                         combined=False)

        ValueByBinnedDistancePlot(
            df, 'density', 'distance to wall (cm)', fig, ax, stat_axs, kind='strip',
            first_plot_title='< {:d} cm from wall'.format(ValueByBinnedDistancePlot.distance_bin_width),
            xlabel='distance to wall (cm)', ylabel='proportion of fields / m$^2$',
            friedman_grouping_variable='animal',
            plot_stats_test='Mann-Whitney',
            aggregate_by_distance_and_animal='mean',
            data_selection_label_kwargs={}
        )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(left=0.16, bottom=0.16, right=0.995, top=0.85, wspace=0.25, hspace=0.4)

        stat_fig, stat_axs = plt.subplots(2, 1, figsize=(10, 25), gridspec_kw={'height_ratios': [2.5, 4]})
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs.flatten():
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        FieldDistributionWithinEnvironments.plot(all_recordings, df_units, df_fields, fig, ax, stat_axs)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldDistributionWithinEnvironments'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldDistributionWithinEnvironments.make_figure(all_recordings, df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class BinnedByDistancePlots:

    @staticmethod
    def compute_and_write_stats(df, binned_measure, value, stat_ax):

        x_bin_order = sorted(df[binned_measure].unique())
        df = df.sort_values('animal')
        test_groups = [df[df[binned_measure] == group][value]
                       for group in x_bin_order]

        kruskal_h_value, kruskal_pvalue = kruskal(*test_groups)
        if all(test_groups[0].size == x.size for x in test_groups[1:]) and len(test_groups) > 2:
            friedman_chisq_value, friedman_pvalue = friedmanchisquare(*test_groups)
        else:
            friedman_chisq_value, friedman_pvalue = (np.nan, np.nan)

        test_results = []
        for test in ('Mann-Whitney', 'Wilcoxon'):
            if (
                    len(test_groups) == 2
                    or (test == 'Mann-Whitney' and kruskal_pvalue <= 0.05)
                    or (test == 'Wilcoxon' and friedman_pvalue <= 0.05)
            ):
                one_test_result_dict_list = compute_pairwise_comparisons(df, binned_measure, value,
                                                                         list(combinations(x_bin_order, 2)), test=test)
                if len(one_test_result_dict_list) > 1:
                    p_values = [test_result_dict['p-value'] for test_result_dict in one_test_result_dict_list]
                    _, p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
                    for test_result_dict, p_value in zip(one_test_result_dict_list, p_values):
                        test_result_dict['p-value'] = p_value
                        test_result_dict['correction'] = 'fdr_bh'

                test_results.append(one_test_result_dict_list)

        table_cell_text = [
            ['', 'statistic', 'p-value'],
            ['Kruskal-Wallis test', '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
            ['Friedman test', '{:.2e}'.format(friedman_chisq_value), '{:.2e}'.format(friedman_pvalue)]
        ]
        if 'n' in df.columns:
            for animal in sorted(df['animal'].unique()):
                tmp = [['', '', ''], ['animal', animal, '']]
                for distance_bin in x_bin_order:
                    tmp.append(['bin', str(distance_bin),
                                'n={}'.format(
                                    df[(df['animal'] == animal)
                                       & (df[binned_measure] == distance_bin)]['n'].values[0])]
                               )
                table_cell_text += tmp

        table_cell_text.append(['', '', ''])
        table_cell_text.append(['bin', 'mean', ''])
        for x_bin in x_bin_order:
            table_cell_text.append([str(x_bin), str(np.nanmean(df.loc[df[binned_measure] == x_bin, value])), ''])

        table_cell_text.append(['', '', ''])
        table_cell_text.append(['Mean overall', str(np.nanmean(df[value])), ''])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')
        plot_stats_dict_to_axes(test_results, stat_ax, loc=(0, 0), va='bottom')

        significance_found = (
                kruskal_pvalue <= 0.05 or friedman_pvalue <= 0.05
                or (len(test_groups) == 2 and any([d['p-value'] <= 0.05 for d in sum(test_results, [])]))
        )
        return significance_found

    @staticmethod
    def compute_and_write_stats_with_hue(df, binned_measure, value, hue, stat_ax):

        hue_values = df[hue].unique()
        x_bin_order = sorted(df[binned_measure].unique())

        if len(hue_values) != 2:
            raise Exception('Only 2 hue levels supported.')

        table_cell_text = []

        for hue_value in hue_values:

            df_tmp = df.loc[df[hue] == hue_value]

            df_tmp = df_tmp.sort_values('animal')
            test_groups = [df_tmp[df_tmp[binned_measure] == group][value]
                           for group in x_bin_order]

            kruskal_h_value, kruskal_pvalue = kruskal(*test_groups)
            if all(test_groups[0].size == x.size for x in test_groups[1:]) and len(test_groups) > 2:
                friedman_chisq_value, friedman_pvalue = friedmanchisquare(*test_groups)
            else:
                friedman_chisq_value, friedman_pvalue = (np.nan, np.nan)

            table_cell_text += [
                ['', 'statistic', 'p-value'],
                ['Kruskal-Wallis test', '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
                ['Friedman test', '{:.2e}'.format(friedman_chisq_value), '{:.2e}'.format(friedman_pvalue)]
            ]

            if 'n' in df_tmp.columns:
                for animal in sorted(df_tmp['animal'].unique()):
                    tmp = [['', '', ''], ['animal', animal, '']]
                    for distance_bin in x_bin_order:
                        tmp.append(['bin', str(distance_bin),
                                    'n={}'.format(
                                        df_tmp[(df_tmp['animal'] == animal)
                                           & (df_tmp[binned_measure] == distance_bin)]['n'].values[0])]
                                   )
                    table_cell_text += tmp

        test_results = []
        for test in ('Mann-Whitney', 'Wilcoxon'):

            for binned_measure_value in df[binned_measure].unique():
                df_tmp = df.loc[df[binned_measure] == binned_measure_value]
                test_results.append(
                    compute_pairwise_comparisons(df_tmp, hue, value,
                                                 [hue_values], test=test)
                )

        table_cell_text.append(['', '', ''])
        table_cell_text.append(['bin', 'mean', ''])
        for x_bin in x_bin_order:
            table_cell_text.append([str(x_bin), str(np.nanmean(df.loc[df[binned_measure] == x_bin, value])), ''])

        table_cell_text.append(['', '', ''])
        table_cell_text.append(['Mean overall', str(np.nanmean(df[value])), ''])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')
        plot_stats_dict_to_axes(test_results, stat_ax, loc=(0, 0), va='bottom')

        significance_found = True
        return significance_found

    @staticmethod
    def plot_value_binned_by_distance_as_scatter_for_one_environment(
        df, binned_measure, bin_edges, value, environment, jitter, ax, stat_ax,
        animal_colors_dict, filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
        orientation_rule=None, direction_rule=None, yscale=None, hue=None, legend=False
    ):
        if legend and hue is None:
            raise Exception('legend only available if hue specified.')

        distance_bin_width = bin_edges[1] - bin_edges[0]

        df = df.loc[df['environment'] == environment]
        bin_edges = bin_edges[bin_edges < df[binned_measure].max() + distance_bin_width]

        if hue is None:
            bin_jitters = np.random.uniform(-jitter, jitter, df[binned_measure].size)
            ax.scatter(
                df[binned_measure] + bin_jitters,
                df[value], s=50, c=[animal_colors_dict[x] for x in df['animal']],
                linewidth=1, edgecolors='black', zorder=1, alpha=0.75
            )
        else:
            hue_values = df[hue].unique()
            if len(hue_values) != 2:
                raise Exception('Hue {} has {} different values, but only exactly 2 is accepted.'.format(
                    hue, len(hue_values)))

            scatter_handles = []
            for hue_value, marker, hue_jitter in zip(hue_values, ('o', 'v'), (-jitter * 1.5, jitter * 1.5)):

                df_hue = df.loc[df[hue] == hue_value]

                bin_jitters = np.random.uniform(-jitter, jitter, df_hue[binned_measure].size) + hue_jitter
                scatter_handles.append(ax.scatter(
                    df_hue[binned_measure] + bin_jitters,
                    df_hue[value], s=50, c=[animal_colors_dict[x] for x in df_hue['animal']],
                    linewidth=1, edgecolors='black', zorder=1, alpha=0.75, marker=marker, label=hue_value
                ))

            if legend:
                ax.legend(title=hue)

        ax.set_xticks(bin_edges)
        ax.set_xlim((bin_edges[0], bin_edges[-1]))

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.set_title('')
        spatial_filter_legend_instance.append_to_axes(
            ax, experiment_id_substitutes_inverse[environment],
            distance_measure=binned_measure,
            distance_bin_width=distance_bin_width,
            filter_to_middle_third=filter_to_middle_third,
            filter_to_first_bin_from_wall=filter_to_first_bin_from_wall,
            orientation_rule=orientation_rule,
            direction_rule=direction_rule,
            proportional_to_environment_size=True,
            max_bin_center=bin_edges[-1]
        )

        # Stats

        if hue is None:
            significance_found = BinnedByDistancePlots.compute_and_write_stats(df, binned_measure, value, stat_ax)
        else:
            significance_found = BinnedByDistancePlots.compute_and_write_stats_with_hue(df, binned_measure, value,
                                                                                        hue, stat_ax)

        if significance_found and yscale is None:
            ylim = list(ax.get_ylim())
            ylim[1] = ylim[1] + 0.5 * (ylim[1] - ylim[0])
            ax.set_ylim(ylim)

    @staticmethod
    def plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, binned_measure, bin_edges, value, xlabel, ylabel,
            filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None, yscale=None, ymax=None,
            plot_first_bin_comparison_between_environments=True, hue=None
    ):

        if plot_first_bin_comparison_between_environments and hue is not None:
            raise Exception('Plotting first bin comparison with hue is not supported')

        environments = sorted(df['environment'].unique())
        df['environment_number'] = df['environment'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4})

        binned_measure_order = {}
        for environment in environments:
            binned_measure_order[environment] = \
                sorted(df.loc[df['environment'] == environment, binned_measure].unique())
        first_bin = list(binned_measure_order.values())[0][0]

        # Create ax

        if plot_first_bin_comparison_between_environments:
            width_ratios = [len(environments), 0.01] + [len(binned_measure_order[env]) for env in environments[1:]]
            gs = GridSpecFromSubplotSpec(1, len(width_ratios), ax, wspace=0.25, width_ratios=width_ratios)
            ax.axis('off')
            axs = [ax.figure.add_subplot(g) for g in gs]
            ax_empty = axs.pop(1)
            ax_empty.axis('off')
            ax_first_bin = axs.pop(0)
        else:
            width_ratios = [len(binned_measure_order[env]) for env in environments[1:]]
            gs = GridSpecFromSubplotSpec(1, len(width_ratios), ax, wspace=0.25, width_ratios=width_ratios)
            ax.axis('off')
            axs = [ax.figure.add_subplot(g) for g in gs]
            ax_first_bin = None

        # Create stat ax

        gs = GridSpecFromSubplotSpec(1, len(axs) + 1, stat_ax)
        stat_ax.axis('off')
        stat_axs = [stat_ax.figure.add_subplot(g) for g in gs]

        # Plot

        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        if plot_first_bin_comparison_between_environments:
            df_first_bin = df.loc[df[binned_measure] == first_bin]
            ax_first_bin.scatter(
                df_first_bin['environment_number'] + np.random.uniform(-0.2, 0.2, df_first_bin['environment_number'].size),
                df_first_bin[value], s=50, c=[colors_dict[x] for x in df_first_bin['animal']],
                linewidth=1, edgecolors='black', zorder=1, alpha=0.75
            )
            ax_first_bin.set_xlim((df_first_bin['environment_number'].min() - 0.5,
                                   df_first_bin['environment_number'].max() + 0.5))
            ax_first_bin.set_xticks(sorted(df_first_bin['environment_number'].unique()))
            ax_first_bin.set_xticklabels(environments)

            if yscale is not None:
                ax_first_bin.set_yscale(yscale)

            significance_found = BinnedByDistancePlots.compute_and_write_stats(
                df_first_bin, 'environment_number', value, stat_axs[0]
            )
            if significance_found and yscale is None:
                ylim = list(ax_first_bin.get_ylim())
                ylim[1] = ylim[1] + 0.5 * (ylim[1] - ylim[0])
                ax_first_bin.set_ylim(ylim)

        for i, environment in enumerate(environments[1:]):

            BinnedByDistancePlots.plot_value_binned_by_distance_as_scatter_for_one_environment(
                df, binned_measure, bin_edges, value, environment, 0.2 * first_bin,
                axs[i], stat_axs[i + 1], colors_dict,
                filter_to_middle_third=filter_to_middle_third,
                filter_to_first_bin_from_wall=filter_to_first_bin_from_wall,
                orientation_rule=orientation_rule, direction_rule=direction_rule, yscale=yscale,
                hue=hue, legend=(True if (hue is not None and environment == 'D') else False)
            )

        if plot_first_bin_comparison_between_environments:
            all_axs = [ax_first_bin] + axs
        else:
            all_axs = axs
        ylim = get_max_ylim(all_axs)
        ylim = ylim if ymax is None else (ylim[0], ymax)
        for ax in all_axs:
            ax.set_ylim(ylim)

        if plot_first_bin_comparison_between_environments:
            for ax in axs:
                ax.set_yticklabels([], [])
        else:
            axs[0].set_ylabel(ylabel)
            for ax in axs[1:]:
                ax.set_yticklabels([], [])

        if plot_first_bin_comparison_between_environments:
            ax_first_bin.set_ylabel(ylabel)
            ax_first_bin.set_xlabel('environment')
            if bin_edges[1] % 1 == 0:
                ax_first_bin.set_title('< {:d} cm from wall'.format(int(bin_edges[1])))
            else:
                ax_first_bin.set_title('< {:.2f} cm from wall'.format(bin_edges[1]))

        axs[1].set_xlabel(xlabel)

        return all_axs


class FieldDensity:

    bin_size = 25
    max_bin_center = 100

    environment_wall_sizes = {experiment_id: np.array([spatial_window[1], spatial_window[3]])
                              for experiment_id, spatial_window in spatial_windows.items()}

    @staticmethod
    def compute_environment_wall_distance_bin_areas(experiment_id):
        large_shape = FieldDensity.environment_wall_sizes[experiment_id]
        current_distance = float(FieldDensity.bin_size) / 2.

        areas = {}
        while True:
            small_shape = large_shape - 2 * FieldDensity.bin_size
            if np.any(small_shape < 0):
                break

            areas[current_distance] = (np.prod(large_shape) - np.prod(small_shape)) / (10 ** 4)
            current_distance = current_distance + FieldDensity.bin_size
            large_shape = small_shape

        return areas

    @staticmethod
    def compute_density_per_bin(df):

        experiment_id = df['experiment_id'].values[0]
        distance_bin_areas = FieldDensity.compute_environment_wall_distance_bin_areas(experiment_id)
        ValueByBinnedDistancePlot.bin_distance_values(df, 'peak_nearest_wall', FieldDensity.bin_size,
                                                      environment_column='experiment_id')
        df.drop(index=df.loc[df['peak_nearest_wall'] > FieldDensity.max_bin_center].index, inplace=True)
        df['count'] = 1
        df = df.groupby(['experiment_id', 'animal', 'peak_nearest_wall']).sum().reset_index()
        df['area'] = [distance_bin_areas[distance_bin] for distance_bin in df['peak_nearest_wall'].values]
        df['density'] = df['count'] / df['area']
        del df['count']
        del df['area']
        del df['experiment_id']
        del df['animal']

        df.rename(columns={'peak_nearest_wall': 'distance to wall (cm)'}, inplace=True)

        return df

    @staticmethod
    def compute(df_units, df_fields):

        # Create a copy of df_fields with only the relevant columns
        df = df_fields.loc[df_fields['experiment_id'] != 'exp_scales_a2',
            ['animal', 'animal_unit', 'experiment_id', 'peak_nearest_wall']
        ].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[df['category'] == 'place_cell']  # Only keep place cell fields
        df = df[['experiment_id', 'animal', 'peak_nearest_wall']]

        df_count_density = \
            df.groupby(['experiment_id', 'animal']).apply(FieldDensity.compute_density_per_bin).reset_index()
        df_count = \
            df.groupby('animal').count()[['experiment_id']].reset_index().rename(columns={'experiment_id': 'count'})
        df = df_count_density.merge(df_count, how='left', on='animal')
        df['density'] = df['density'] / df['count']
        del df['count']

        df['environment'] = df['experiment_id'].map(experiment_id_substitutes)

        return df

    @staticmethod
    def plot(df_units, df_fields, ax, stat_ax):
        df = FieldDensity.compute(df_units, df_fields)

        binned_measure = 'distance to wall (cm)'
        value = 'density'
        xlabel = 'distance to wall (cm)'
        ylabel = 'proportion of fields / m$^2$'

        bin_edges = np.arange(0, df[binned_measure].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, binned_measure, bin_edges, value, xlabel, ylabel,
            filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None
        )

    @staticmethod
    def make_figure(df_units, df_fields):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(left=0.14, bottom=0.16, right=0.97, top=0.8, wspace=0.25, hspace=0.4)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(30, 15))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FieldDensity.plot(df_units, df_fields, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldDensity'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldDensity.make_figure(df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldDensityByDwell:

    bin_size = 25
    max_bin_centers = {'A': 25, 'B': 50, 'C': 75, 'D': 100}

    @staticmethod
    def compute_sum_in_bins_within_and_across_distances_to_wall(experiment_id, x, y, values):
        x_width = spatial_windows[experiment_id][1]
        y_width = spatial_windows[experiment_id][3]

        if experiment_id == 'exp_scales_a':
            print('')

        wall_distances = []
        sums = []
        x_centers = []
        y_centers = []
        for x_edge in np.arange(0, x_width, FieldDensityByDwell.bin_size):
            for y_edge in np.arange(0, y_width, FieldDensityByDwell.bin_size):

                x_center = x_edge + FieldDensityByDwell.bin_size / 2.
                y_center = y_edge + FieldDensityByDwell.bin_size / 2.
                x_centers.append(x_center)
                y_centers.append(y_center)
                x_distance = min(x_center, max(x_width - x_center, x_width / 2.))
                y_distance = min(y_center, max(y_width - y_center, x_width / 2.))
                wall_distances.append(min(x_distance, y_distance))

                idx = (((x >= x_edge) & (x < x_edge + FieldDensityByDwell.bin_size))
                       & ((y >= y_edge) & (y < y_edge + FieldDensityByDwell.bin_size)))
                if np.sum(idx) == 0:
                    sums.append(0)
                else:
                    sums.append(np.sum(values[idx]))

        return np.array(wall_distances), np.array(sums), np.array(x_centers), np.array(y_centers)

    @staticmethod
    def get_sampling_maps(all_recordings):

        sampling_maps = {}
        position_bin_centers = {}
        for recordings in all_recordings:

            animal_sampling_maps = {}
            animal_position_bins = {}
            for recording in recordings[:4]:

                spatial_ratemap = SpatialRatemap(
                    recording.position['xy'], np.array([0.1]), recording.position['sampling_rate'],
                    spatial_window=(0, recording.info['arena_size'][0], 0, recording.info['arena_size'][1]),
                    xy_mask=recording.position['analysis']['ratemap_speed_mask'],
                    bin_size=Params.spatial_ratemap['bin_size']
                )

                animal_sampling_maps[recording.info['experiment_id']] = spatial_ratemap.dwell_time

                x_position_bin_centers, y_position_bin_centers = spatial_ratemap.position_bins
                animal_position_bins[recording.info['experiment_id']] = {'x': x_position_bin_centers,
                                                                         'y': y_position_bin_centers}

            sampling_maps[recordings[0].info['animal']] = animal_sampling_maps
            position_bin_centers[recordings[0].info['animal']] = animal_position_bins

        return sampling_maps, position_bin_centers

    @staticmethod
    def compute(all_recordings, df_units, df_fields):

        # Compute field counts

        # Only keep fields belonging to place cells
        df_fields = df_fields[df_fields['unit'].isin(np.where(df_units['category'] == 'place_cell')[0])]
        # Only keep fields not in exp_scales_a2
        df_fields = df_fields[df_fields['experiment_id'] != 'exp_scales_a2']

        dfs = []
        for animal in df_fields['animal'].unique():
            for experiment_id in df_fields['experiment_id'].unique():
                idx = (df_fields['animal'] == animal) & (df_fields['experiment_id'] == experiment_id)
                wall_distances, field_counts, x_centers, y_centers = \
                    FieldDensityByDwell.compute_sum_in_bins_within_and_across_distances_to_wall(
                        experiment_id, df_fields.loc[idx, 'peak_x'], df_fields.loc[idx, 'peak_y'], np.ones(np.sum(idx))
                    )
                dfs.append(pd.DataFrame({
                    'animal': animal, 'experiment_id': experiment_id, 'wall_distance': wall_distances,
                    'x_center': x_centers, 'y_center': y_centers, 'field_count': field_counts
                }))

        df = pd.concat(dfs, axis=0, ignore_index=True)

        # Compute field count proportional to total in animal
        df = df.merge(df_fields.groupby(['animal'])['experiment_id'].count().reset_index().rename(
            columns={'experiment_id': 'animal_total_field_count'}), on='animal', how='left')
        df['proportional_field_count'] = df['field_count'] / df['animal_total_field_count']

        # Compute bin_area
        dfs = []
        for animal in df_fields['animal'].unique():
            for experiment_id in df_fields['experiment_id'].unique():

                x_centers = np.arange(0.5, spatial_windows[experiment_id][1], 1)
                y_centers = np.arange(0.5, spatial_windows[experiment_id][3], 1)
                x_centers, y_centers = np.meshgrid(x_centers, y_centers)
                x_centers = x_centers.flatten()
                y_centers = y_centers.flatten()

                wall_distances, bin_areas_cm, x_centers, y_centers = \
                    FieldDensityByDwell.compute_sum_in_bins_within_and_across_distances_to_wall(
                        experiment_id, x_centers, y_centers, np.ones(x_centers.size)
                    )
                bin_areas_m = bin_areas_cm / (100 ** 2)
                dfs.append(pd.DataFrame({
                    'animal': animal, 'experiment_id': experiment_id, 'wall_distance': wall_distances,
                    'x_center': x_centers, 'y_center': y_centers, 'bin_area': bin_areas_m
                }))

        df = df.merge(pd.concat(dfs, axis=0, ignore_index=True),
                      on=['animal', 'experiment_id', 'wall_distance', 'x_center', 'y_center'],
                      how='left')

        df['proportional_field_density'] = df['proportional_field_count'] / df['bin_area']

        # Compute bin sampling density

        sampling_maps, position_bin_centers = FieldDensityByDwell.get_sampling_maps(all_recordings)

        dfs = []
        for animal in df_fields['animal'].unique():
            for experiment_id in df_fields['experiment_id'].unique():

                x_centers, y_centers = np.meshgrid(position_bin_centers[animal][experiment_id]['x'],
                                                   position_bin_centers[animal][experiment_id]['y'])
                x_centers = x_centers.flatten()
                y_centers = y_centers.flatten()

                wall_distances, dwell_times, x_centers, y_centers = \
                    FieldDensityByDwell.compute_sum_in_bins_within_and_across_distances_to_wall(
                        experiment_id, x_centers, y_centers, sampling_maps[animal][experiment_id].flatten()
                    )
                dfs.append(pd.DataFrame({
                    'animal': animal, 'experiment_id': experiment_id, 'wall_distance': wall_distances,
                    'x_center': x_centers, 'y_center': y_centers, 'dwell_time': dwell_times
                }))

        df = df.merge(pd.concat(dfs, axis=0, ignore_index=True),
                      on=['animal', 'experiment_id', 'wall_distance', 'x_center', 'y_center'],
                      how='left')

        # Split animal and experiment distance bins by dwell time
        dfs = []
        for animal in df['animal'].unique():
            for experiment_id in df['experiment_id'].unique():
                for wall_distance in df['wall_distance'].unique():
                    idx = ((df['animal'] == animal)
                           & (df['experiment_id'] == experiment_id)
                           & (df['wall_distance'] == wall_distance))
                    dwell_times = df.loc[idx, 'dwell_time'].values
                    dwell_time_group = np.array(['low' for _ in range(dwell_times.size)], dtype=np.object)
                    dwell_time_group[dwell_times >= np.median(dwell_times)] = 'high'

                    dfs.append(pd.DataFrame({
                        'animal': animal, 'experiment_id': experiment_id, 'wall_distance': wall_distance,
                        'x_center': df.loc[idx, 'x_center'].values, 'y_center': df.loc[idx, 'y_center'].values,
                        'dwell': dwell_time_group
                    }))

        df = df.merge(pd.concat(dfs, axis=0, ignore_index=True),
                      on=['animal', 'experiment_id', 'wall_distance', 'x_center', 'y_center'],
                      how='left')

        df = df.groupby(['animal', 'experiment_id', 'wall_distance', 'dwell'])['proportional_field_density'].mean()
        df = df.reset_index()

        df.rename(columns={'wall_distance': 'distance to wall (cm)'}, inplace=True)
        df['environment'] = df['experiment_id'].map(experiment_id_substitutes)

        for environment in df['environment'].unique():
            df.drop(index=df.loc[(df['distance to wall (cm)'] > FieldDensityByDwell.max_bin_centers[environment])
                                 & (df['environment'] == environment)].index, inplace=True)

        return df

    @staticmethod
    def plot(all_recordings, df_units, df_fields, ax, stat_ax):

        df = FieldDensityByDwell.compute(all_recordings, df_units, df_fields)

        binned_measure = 'distance to wall (cm)'
        value = 'proportional_field_density'
        xlabel = 'distance to wall (cm)'
        ylabel = 'proportion of fields / m$^2$'
        hue = 'dwell'

        bin_edges = np.arange(0, df[binned_measure].max() + FieldDensityByDwell.bin_size, FieldDensityByDwell.bin_size)

        BinnedByDistancePlots.plot_value_binned_by_distance_as_scatter_for_one_environment(
            df.loc[df['environment'] == 'D'], binned_measure, bin_edges, value, 'D', 0.1 * FieldDensityByDwell.bin_size,
            ax, stat_ax, {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)},
            filter_to_middle_third=False,
            filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None,
            hue=hue, legend=True
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
        #     df, ax, stat_ax, binned_measure, bin_edges, value, xlabel, ylabel,
        #     filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
        #     orientation_rule=None, direction_rule=None,
        #     plot_first_bin_comparison_between_environments=False, hue=hue
        # )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields):

        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        plt.subplots_adjust(left=0.28, bottom=0.16, right=0.93, top=0.82, wspace=0.25, hspace=0.4)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(30, 15))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FieldDensityByDwell.plot(all_recordings, df_units, df_fields, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldDensityByDwell'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldDensityByDwell.make_figure(all_recordings, df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldSize(object):
    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def plot_field_size_by_distance_to_wall(all_recordings, df_units, df_fields, ax, stat_ax, verbose=False):
        """Plotted data:
        - field area
        - by distance to closest wall
        - including all positions
        """

        if verbose:
            print('Plotting field size by distance to wall')

        df = get_field_data_with_distance_to_boundary(
            all_recordings, df_units, df_fields, 'area', FieldSize.experiment_ids, verbose=verbose
        )

        # df['value'] = np.log10(df['value'])

        ValueByBinnedDistancePlot.bin_distance_values(df, 'distance to wall (cm)', FieldDensity.bin_size)
        df.drop(index=df.loc[df['distance to wall (cm)'] > FieldDensity.max_bin_center].index, inplace=True)

        df = df[['animal', 'environment', 'distance to wall (cm)', 'value']]

        df = df.groupby(['animal', 'environment', 'distance to wall (cm)']).mean().reset_index()

        bin_edges = np.arange(0, df['distance to wall (cm)'].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, 'distance to wall (cm)', bin_edges, 'value',
            'distance to wall (cm)', 'field area (cm$^2$)',
            filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None, yscale='log', ymax=(10 ** 5)
        )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields, verbose=False):

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plt.subplots_adjust(left=0.14, bottom=0.16, right=0.97, top=0.8, wspace=0.25, hspace=0.4)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(30, 15))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FieldSize.plot_field_size_by_distance_to_wall(all_recordings, df_units, df_fields, ax, stat_ax,
                                                      verbose=verbose)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldSize'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldSize.make_figure(all_recordings, df_units, df_fields, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldWidth(object):

    @staticmethod
    def plot_width_in_one_orientation_to_wall(df, orientation, ax, stat_ax):

        if orientation == 'orthogonal':
            values = 'width_orthogonal_to_short_wall'
            orientation_rule = 'orthogonal_to_short_wall'
            ylabel = r'field width $\bf\bot$ to short wall (cm)'
        elif orientation == 'parallel':
            values = 'width_parallel_to_short_wall'
            orientation_rule = 'parallel_to_short_wall'
            ylabel = r'field width $\bf\parallel$ to short wall (cm)'
        else:
            raise ValueError('Unknown orientation: {}'.format(orientation))

        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        distance_bins = sorted(np.unique(df['distance to short wall (cm)']))
        bin_edges = np.arange(0, df['distance to short wall (cm)'].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        BinnedByDistancePlots.plot_value_binned_by_distance_as_scatter_for_one_environment(
            df, 'distance to short wall (cm)', bin_edges,
            values, 'D', 0.1 * np.max(np.diff(distance_bins)), ax, stat_ax, colors_dict,
            filter_to_middle_third=True,
            filter_to_first_bin_from_wall=False,
            orientation_rule=orientation_rule, direction_rule=None
        )

        ax.set_xlabel('distance to short wall (cm)')
        ax.set_ylabel(ylabel)

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields, verbose=False):

        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        plt.subplots_adjust(left=0.18, bottom=0.16, right=0.95, top=0.8, wspace=0.5)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(20, 20))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs.flatten():
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        binned_measure = 'distance to short wall (cm)'
        measures = ['width_parallel_to_short_wall', 'width_orthogonal_to_short_wall']
        df = get_field_data_with_distance_to_boundary(
            all_recordings, df_units, df_fields, measures,
            FieldSize.experiment_ids, verbose=verbose
        )
        df = df.loc[df['environment'] == 'D'].copy()
        df.dropna(inplace=True)

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure,
                                                      FieldDensity.bin_size)
        df.drop(index=df.loc[df[binned_measure] > FieldDensity.max_bin_center].index, inplace=True)
        df = ValueByBinnedDistancePlot.filter_to_middle_third(df)
        df = df[['animal', 'environment', binned_measure] + measures]
        agg_columns = ['animal', 'environment', binned_measure]
        df = df.groupby(agg_columns).mean().reset_index()

        FieldWidth.plot_width_in_one_orientation_to_wall(df, 'orthogonal', axs[0], stat_axs[0])
        FieldWidth.plot_width_in_one_orientation_to_wall(df, 'parallel', axs[1], stat_axs[1])

        ylim = get_max_ylim(axs)
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldWidth'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldWidth.make_figure(all_recordings, df_units, df_fields, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class AverageActivity:

    @staticmethod
    def plot_one_value_binned_by_distance_to_wall_in_one_environment(df, values, ylabel, ax, stat_ax):

        ValueByBinnedDistancePlot.bin_distance_values(df, 'distance to wall (cm)',
                                                      FieldDensity.bin_size)
        df.drop(index=df.loc[df['distance to wall (cm)'] > FieldDensity.max_bin_center].index, inplace=True)

        df = df.loc[df['environment'] == 'D', ['animal', 'environment', 'distance to wall (cm)', values]].copy()
        agg_columns = ['animal', 'environment', 'distance to wall (cm)']
        df = df.groupby(agg_columns).mean().reset_index()

        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        distance_bins = sorted(np.unique(df['distance to wall (cm)']))
        bin_edges = np.arange(0, df['distance to wall (cm)'].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        BinnedByDistancePlots.plot_value_binned_by_distance_as_scatter_for_one_environment(
            df, 'distance to wall (cm)', bin_edges,
            values, 'D', 0.1 * np.max(np.diff(distance_bins)), ax, stat_ax, colors_dict,
            filter_to_middle_third=False,
            filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None
        )

        ax.set_xlabel('distance to wall (cm)')
        ax.set_ylabel(ylabel)

    @staticmethod
    def make_figure(all_recordings):

        fig, axs = plt.subplots(1, 2, figsize=(6, 4))
        plt.subplots_adjust(left=0.13, bottom=0.16, right=0.95, top=0.8, wspace=0.5)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(20, 15))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs.flatten():
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        AverageActivity.plot_one_value_binned_by_distance_to_wall_in_one_environment(
            FieldCoverage.get_mean_ratemap_measure_for_each_position(
                all_recordings, 'active_unit_count', mean_per_field=False
            ),
            'value', 'proportion of cells co-active', axs[0], stat_axs[0]
        )
        axs[0].set_ylim(0, 0.4)
        AverageActivity.plot_one_value_binned_by_distance_to_wall_in_one_environment(
            FieldCoverage.get_mean_ratemap_measure_for_each_position(
                all_recordings, 'firing_rate', mean_per_field=False,
            ),
            'value', 'mean spike rate (Hz)', axs[1], stat_axs[1]
        )
        axs[1].set_ylim(0, 1.2)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, prefix='', verbose=True):

        figure_name = prefix + 'AverageActivity'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = AverageActivity.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class InterneuronMeanRate(object):

    @staticmethod
    def get_dataframe(all_recordings, min_speed: float = 10, unit_category: str = 'interneuron',
                      experiment_id: str = 'exp_scales_d', verbose: bool = True):

        active_firing_rate = []
        x_coords = []
        y_coords = []
        distances = []
        animals = []
        experiment_ids = []
        environments = []
        movement_directions = []
        for recordings in all_recordings:
            for i_recording, recording in enumerate(recordings[:4]):

                if experiment_id is not None and recording.info['experiment_id'] != experiment_id:
                    continue

                if verbose:
                    print('Computing {} mean rate for animal {} experiment {}'.format(
                        unit_category, recording.info['animal'], recording.info['experiment_id']
                    ))

                idx_samples_in_environment = (
                    np.all(recording.position['xy'] > 0, axis=1)
                    & (recording.position['xy'][:, 0] <= recording.info['arena_size'][0])
                    & (recording.position['xy'][:, 1] <= recording.info['arena_size'][1])
                )
                idx_samples_with_sufficent_speed = recording.position['speed'] > min_speed
                idx_position_samples_to_use = idx_samples_in_environment & idx_samples_with_sufficent_speed

                # Compute population vector
                population_vectors = []
                for i_unit, recordings_unit in enumerate(recordings.units):
                    if recordings.first_available_recording_unit(i_unit)['analysis']['category'] != unit_category:
                        continue

                    unit = recordings_unit[i_recording]

                    if unit is None:
                        timestamps = np.array([1])
                    else:
                        timestamps = unit['timestamps']

                    spike_histogram = count_spikes_in_sample_bins(
                        timestamps, recording.position['sampling_rate'],
                        0, recording.position['xy'].shape[0] - 1,
                        sum_samples=9,
                        sum_samples_kernel='gaussian'
                    )
                    spike_histogram *= 0 if unit is None else recording.position['sampling_rate']

                    population_vectors.append(spike_histogram)

                population_vectors = np.stack(population_vectors, axis=1)

                population_vectors = population_vectors[idx_position_samples_to_use, :]
                sample_xy = recording.position['xy'][idx_position_samples_to_use, :]

                # Compute movement direction
                movement_direction = Recording.compute_movement_direction(sample_xy)

                # Append to list across animals and recordings
                active_firing_rate.append(np.mean(population_vectors, axis=1))
                x_coords.append(sample_xy[:, 0])
                y_coords.append(sample_xy[:, 1])
                distances.append(np.array([
                    snippets.compute_distance_to_nearest_wall(one_sample_xy, recording.info['arena_size'])
                    for one_sample_xy in list(sample_xy)
                ]))
                animals.append(np.array([recording.info['animal']] * sample_xy.shape[0]))
                experiment_ids.append(np.array([recording.info['experiment_id']] * sample_xy.shape[0]))
                environments.append(
                    np.array([experiment_id_substitutes[recording.info['experiment_id']]] * sample_xy.shape[0]))
                movement_directions.append(movement_direction)

        df = pd.DataFrame({
            'animal': np.concatenate(animals),
            'environment': np.concatenate(environments),
            'experiment_id': np.concatenate(experiment_ids),
            'x_coord': np.concatenate(x_coords),
            'y_coord': np.concatenate(y_coords),
            'distance to wall (cm)': np.concatenate(distances),
            'mean rate (Hz)': np.concatenate(active_firing_rate),
            'direction': np.concatenate(movement_directions)
        })

        compute_distances_to_landmarks(df, np.stack((df['x_coord'].values, df['y_coord'].values), axis=1))

        return df

    @staticmethod
    def plot_all(all_recordings, ax, stat_ax):

        df = InterneuronMeanRate.get_dataframe(all_recordings)

        AverageActivity.plot_one_value_binned_by_distance_to_wall_in_one_environment(
            df, 'mean rate (Hz)', 'interneuron mean spike rate (Hz)', ax, stat_ax
        )
        ax.set_ylim(0, 40)

    @staticmethod
    def make_figure(all_recordings):

        fig, ax = plt.subplots(1, 1, figsize=(3.5, 4))
        plt.subplots_adjust(left=0.25, bottom=0.16, right=0.9, top=0.8, wspace=0.5)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        InterneuronMeanRate.plot_all(all_recordings, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, prefix='', verbose=True):

        figure_name = prefix + 'InterneuronMeanRate'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = InterneuronMeanRate.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldWidthAll(object):
    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def plot_field_size_by_distance_to_wall(all_recordings, df_units, df_fields, measure, binned_measure,
                                            orientation_rule, xlabel, ylabel, ax, stat_ax, verbose=False):

        df = get_field_data_with_distance_to_boundary(
            all_recordings, df_units, df_fields, measure,
            FieldWidthAll.experiment_ids, verbose=verbose
        )

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure, FieldDensity.bin_size)
        df.drop(index=df.loc[df[binned_measure] > FieldDensity.max_bin_center].index, inplace=True)
        df = ValueByBinnedDistancePlot.filter_to_middle_third(df)

        df = df[['animal', 'environment', binned_measure, 'value']]

        df = df.groupby(['animal', 'environment', binned_measure]).mean().reset_index()

        bin_edges = np.arange(0, df[binned_measure].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        return BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, binned_measure, bin_edges, 'value', xlabel, ylabel,
            filter_to_middle_third=True, filter_to_first_bin_from_wall=False,
            orientation_rule=orientation_rule, direction_rule=None,
            plot_first_bin_comparison_between_environments=False
        )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields, verbose=False):

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.subplots_adjust(left=0.08, bottom=0.16, right=0.98, top=0.86, wspace=0.25, hspace=0.4)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(40, 15))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs:
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        sub_axs_1 = FieldWidthAll.plot_field_size_by_distance_to_wall(
            all_recordings, df_units, df_fields, 'width_parallel_to_long_wall',
            'distance to short wall (cm)', 'orthogonal_to_short_wall',
            'distance to short wall (cm)', r'field width $\bf\bot$ to short wall (cm)',
            axs[0], stat_axs[0], verbose=verbose
        )

        sub_axs_2 = FieldWidthAll.plot_field_size_by_distance_to_wall(
            all_recordings, df_units, df_fields, 'width_parallel_to_short_wall',
            'distance to short wall (cm)', 'parallel_to_short_wall',
            'distance to short wall (cm)', r'field width $\bf\parallel$ to short wall (cm)',
            axs[1], stat_axs[1], verbose=verbose
        )

        all_axs = sub_axs_1 + sub_axs_2
        ylim = get_max_ylim(all_axs)
        for ax in all_axs:
            ax.set_ylim(ylim)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldWidthAll'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldWidthAll.make_figure(all_recordings, df_units, df_fields, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class AverageActivityAll(object):
    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def plot_activity_by_distance_to_wall(all_recordings, df_units, df_fields, measure, binned_measure,
                                          xlabel, ylabel, ax, stat_ax, verbose=False):

        df = FieldCoverage.get_mean_ratemap_measure_for_each_position(all_recordings, measure, mean_per_field=False)

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure, FieldDensity.bin_size)
        df.drop(index=df.loc[df[binned_measure] > FieldDensity.max_bin_center].index, inplace=True)
        df = ValueByBinnedDistancePlot.filter_to_middle_third(df)

        df = df[['animal', 'environment', binned_measure, 'value']]

        df = df.groupby(['animal', 'environment', binned_measure]).mean().reset_index()

        bin_edges = np.arange(0, df[binned_measure].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        return BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, binned_measure, bin_edges, 'value', xlabel, ylabel,
            filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=None,
            plot_first_bin_comparison_between_environments=True
        )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields, verbose=False):

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(left=0.07, bottom=0.16, right=0.98, top=0.84, wspace=0.25, hspace=0.3)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(40, 15))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs:
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        sub_axs = AverageActivityAll.plot_activity_by_distance_to_wall(
            all_recordings, df_units, df_fields, 'active_unit_count',
            'distance to wall (cm)', 'distance to wall (cm)', 'proportion of cells co-active',
            axs[0], stat_axs[0], verbose=verbose
        )
        for ax in sub_axs:
            ax.set_ylim((0, 0.4))

        sub_axs = AverageActivityAll.plot_activity_by_distance_to_wall(
            all_recordings, df_units, df_fields, 'firing_rate',
            'distance to wall (cm)', 'distance to wall (cm)', 'mean spike rate (Hz)',
            axs[1], stat_axs[1], verbose=verbose
        )
        for ax in sub_axs:
            ax.set_ylim((0, 1.2))

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'AverageActivityAll'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = AverageActivityAll.make_figure(all_recordings, df_units, df_fields, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FiringRateDistribution(object):

    @staticmethod
    def compute(all_recordings,
                only_take_every_nth_sample: int = 15,
                min_speed: float = 10,
                min_spike_rate: float = 1.0,
                verbose: bool = True):

        smoothing_position_samples = 15
        smoothing_method = 'gaussian'

        x_coords = []
        y_coords = []
        distances = []
        directions = []
        spike_rates = []
        animals = []
        experiment_ids = []
        environments = []
        for recordings in all_recordings:
            for i_recording, recording in enumerate(recordings[:4]):

                if recording.info['experiment_id'] != 'exp_scales_d':
                    continue

                if verbose:
                    print('Computing firing rates for animal {} experiment {}'.format(
                        recording.info['animal'], recording.info['experiment_id']
                    ))

                inds = None
                if only_take_every_nth_sample is None:
                    xy = recording.position['xy']
                    speed = recording.position['speed']
                    movement_direction = Recording.compute_movement_direction(xy)
                else:
                    inds = np.arange(0, recording.position['xy'].shape[0], only_take_every_nth_sample)
                    xy = recording.position['xy'][inds, :]
                    speed = recording.position['speed'][inds]
                    movement_direction = Recording.compute_movement_direction(recording.position['xy'])[inds]

                idx_samples_in_environment = (
                        np.all(xy > 0, axis=1)
                        & (xy[:, 0] <= recording.info['arena_size'][0])
                        & (xy[:, 1] <= recording.info['arena_size'][1])
                )
                idx_samples_with_sufficent_speed = speed > min_speed
                idx_position_samples_to_use = idx_samples_in_environment & idx_samples_with_sufficent_speed

                xy = xy[idx_position_samples_to_use, :]
                movement_direction = movement_direction[idx_position_samples_to_use]

                distance = np.array([
                    snippets.compute_distance_to_nearest_wall(one_sample_xy, recording.info['arena_size'])
                    for one_sample_xy in list(xy)
                ])

                animal = np.array([recording.info['animal']] * xy.shape[0])
                experiment_id = np.array([recording.info['experiment_id']] * xy.shape[0])
                environment = np.array([experiment_id_substitutes[recording.info['experiment_id']]] * xy.shape[0])

                # Compute population vector
                for i_unit, recordings_unit in enumerate(recordings.units):
                    if recordings.first_available_recording_unit(i_unit)['analysis']['category'] != 'place_cell':
                        continue

                    unit = recordings_unit[i_recording]

                    if unit is None:
                        timestamps = np.array([1])
                    else:
                        timestamps = unit['timestamps']

                    spike_histogram = count_spikes_in_sample_bins(
                        timestamps, recording.position['sampling_rate'],
                        0, recording.position['xy'].shape[0] - 1,
                        sum_samples=smoothing_position_samples,
                        sum_samples_kernel=smoothing_method
                    )
                    spike_histogram *= 0 if unit is None else recording.position['sampling_rate']

                    if only_take_every_nth_sample is None:
                        spike_rates.append(spike_histogram[idx_position_samples_to_use])
                    else:
                        spike_rates.append(spike_histogram[inds][idx_position_samples_to_use])

                    x_coords.append(xy[:, 0])
                    y_coords.append(xy[:, 1])
                    distances.append(distance)
                    directions.append(movement_direction)
                    animals.append(animal)
                    experiment_ids.append(experiment_id)
                    environments.append(environment)

        df = pd.DataFrame({
            'animal': np.concatenate(animals),
            'environment': np.concatenate(environments),
            'experiment_id': np.concatenate(experiment_ids),
            'spike rate (Hz)': np.concatenate(spike_rates),
            'x_coord': np.concatenate(x_coords),
            'y_coord': np.concatenate(y_coords),
            'distance to wall (cm)': np.concatenate(distances),
            'direction': np.concatenate(directions)
        })
        df = df.loc[df['spike rate (Hz)'] >= min_spike_rate].copy()

        df['log10( spike rate (Hz) )'] = np.log10(df['spike rate (Hz)'])

        # compute_distances_to_landmarks(df, np.stack((df['x_coord'].values, df['y_coord'].values), axis=1))

        return df

    @staticmethod
    def label_yaxis_as_log10(ax):
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))

        yticklabels = ['10$^{:d}$'.format(int(y)) for y in ax.get_yticks(False)]

        ax.set_yticklabels([], minor=True)
        ax.set_yticklabels(yticklabels, minor=False)

    @staticmethod
    def plot(all_recordings, axs, stat_ax):

        df = FiringRateDistribution.compute(all_recordings)

        ValueByBinnedDistancePlot.bin_distance_values(df, 'distance to wall (cm)',
                                                      FieldDensity.bin_size)
        df.drop(index=df.loc[df['distance to wall (cm)'] > FieldDensity.max_bin_center].index, inplace=True)

        bin_edges = np.arange(0, df['distance to wall (cm)'].max() + FieldDensity.bin_size, FieldDensity.bin_size)
        if np.all([x % 1 == 0 for x in bin_edges]):
            bin_edges = bin_edges.astype(np.int32)

        df['distance to wall (cm)'] = list(map(str, df['distance to wall (cm)'].astype(np.int16)))
        distance_bins = sorted(df['distance to wall (cm)'].unique())

        df['recording halves'] = ''

        for animal in sorted(df['animal'].unique()):
            for distance_to_wall in distance_bins:
                indices = df.loc[(df['distance to wall (cm)'] == distance_to_wall) & (df['animal'] == animal)].index
                first_half_index_count = int(round(indices.size / 2.))
                df.loc[indices[:first_half_index_count], 'recording halves'] = '1st'
                df.loc[indices[first_half_index_count:], 'recording halves'] = '2nd'

        sns.violinplot(x='distance to wall (cm)', y='log10( spike rate (Hz) )', hue='recording halves', data=df,
                       split=False, scale='width', ax=axs[0],
                       palette=sns.color_palette(sns_other_colors[:2], n_colors=2))

        xticks = axs[0].get_xticks()
        xtick_spacing = xticks[1] - xticks[0]
        xticks = xticks - xtick_spacing / 2.
        xticks = np.append(xticks, xticks[-1] + xtick_spacing)
        axs[0].set_xticks(xticks)
        axs[0].set_xticklabels(list(map(str, bin_edges)))

        axs[0].set_ylim((-0.25, 2.25))
        FiringRateDistribution.label_yaxis_as_log10(axs[0])
        axs[0].set_ylabel('active place cell\nspike rate (Hz)')

        distribution_bin_edges = np.linspace(0, df['log10( spike rate (Hz) )'].max(), 100)

        groups = []
        distances = []
        for distance_to_wall in distance_bins:
            groups.append('different recording halves')

            first_half_distribution, _ = np.histogram(
                df.loc[(df['distance to wall (cm)'] == distance_to_wall)
                       & (df['recording halves'] == '1st'), 'log10( spike rate (Hz) )'],
                distribution_bin_edges
            )
            first_half_distribution = first_half_distribution / np.sum(first_half_distribution)
            second_half_distribution, _ = np.histogram(
                df.loc[(df['distance to wall (cm)'] == distance_to_wall)
                       & (df['recording halves'] == '2nd'), 'log10( spike rate (Hz) )'],
                distribution_bin_edges
            )
            second_half_distribution = second_half_distribution / np.sum(second_half_distribution)
            distances.append(
                jensenshannon(first_half_distribution, second_half_distribution)
            )

        for i, bin_i in enumerate(distance_bins[:-1]):
            for bin_j in distance_bins[i + 1:]:
                for halves in [('1st', '1nd'), ('1st', '2nd'), ('2nd', '1st'), ('2nd', '2nd')]:
                    groups.append('different distances to wall')

                    first_distribution, _ = np.histogram(
                        df.loc[(df['distance to wall (cm)'] == bin_i) & (df['recording halves'] == halves[0]),
                               'log10( spike rate (Hz) )'],
                        distribution_bin_edges
                    )
                    first_distribution = first_distribution / np.sum(first_distribution)
                    second_distribution, _ = np.histogram(
                        df.loc[(df['distance to wall (cm)'] == bin_j) & (df['recording halves'] == halves[1]),
                               'log10( spike rate (Hz) )'],
                        distribution_bin_edges
                    )
                    second_distribution = second_distribution / np.sum(second_distribution)
                    distances.append(
                        jensenshannon(first_distribution, second_distribution)
                    )

        df_distances = pd.DataFrame({'': groups, 'Jensen–Shannon divergence': distances})

        sns.swarmplot(x='Jensen–Shannon divergence', y='', data=df_distances, ax=axs[1],
                      palette=sns.color_palette(sns_other_colors[2:4], n_colors=2))

        statistic, pvalue = mannwhitneyu(
            df_distances.loc[df_distances[''] == 'different recording halves', 'Jensen–Shannon divergence'],
            df_distances.loc[df_distances[''] == 'different distances to wall', 'Jensen–Shannon divergence']
        )

        axs[1].text(
            (df_distances['Jensen–Shannon divergence'].max() + axs[1].get_xlim()[1]) / 2., 0.5,
            'ns' if pvalue > 0.05 else 'p = {:.3f}'.format(pvalue),
            va='center', ha='center'
        )

        table_cell_text = [['Mann-Whitney', 'U statistic', 'p-value'],
                           ['', '{:.3e}'.format(statistic), '{:.3e}'.format(pvalue)]]
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def make_figure(all_recordings):

        fig, axs = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [4, 1]})
        plt.subplots_adjust(left=0.16, bottom=0.13, right=0.97, top=0.95, hspace=0.5)

        axs[1].axis('off')
        gs = GridSpecFromSubplotSpec(1, 2, axs[1], width_ratios=[1, 3])
        ax_bottom = fig.add_subplot(gs[1])

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1.5)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FiringRateDistribution.plot(all_recordings, [axs[0], ax_bottom], stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, prefix='', verbose=True):

        figure_name = prefix + 'FiringRateDistribution'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FiringRateDistribution.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FieldAreaDistribution(object):
    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def compute_field_area_distribution(df_units, df_fields):

        # Create a copy of df_fields with only the relevant columns
        df = df_fields[['animal', 'animal_unit', 'experiment_id', 'area']].copy(deep=True)
        df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[df['category'] == 'place_cell']  # Only keep place cell fields
        df = df[['experiment_id', 'animal', 'area']]

        # Drop fields in exp_scales_a2
        df = df[df['experiment_id'] != 'exp_scales_a2']

        # Get total field area per animal
        total_field_area = {}
        for animal in sorted(df['animal'].unique()):
            total_field_area[animal] = df.loc[df['animal'] == animal, 'area'].sum()

        # Replace experiment_id values for plotting and rename the column
        df.replace(to_replace={'experiment_id': experiment_id_substitutes}, inplace=True)
        df.rename(columns={'experiment_id': 'environment'}, inplace=True)

        # Compute total field area in each environment for each animal
        df = df.groupby(['animal', 'environment']).sum()
        df = df.sort_values('animal')
        df.reset_index(inplace=True)

        # Compute total field area in each environment as percentage of total field area per animal
        df['area'] = df['area'] / df.groupby(['animal'])['area'].transform('sum')

        # Convert environment labels to environment area values
        df['environment'] = np.array([arena_areas_meters_short_env[x] for x in df['environment']])

        # Compute field area as percentage per square metre
        df['area per m2'] = df['area'] / df['environment']

        return df

    @staticmethod
    def plot_field_area_distribution(df_units, df_fields, ax, stat_ax):

        df = FieldAreaDistribution.compute_field_area_distribution(df_units, df_fields)

        # Create variables for plotting
        environments = sorted(np.unique(df['environment']))
        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        # Plot field area distribution
        ax.scatter(df['environment'] + np.random.uniform(-0.3, 0.3, df['environment'].size),
                   df['area'], s=50, c=[colors_dict[x] for x in df['animal']],
                   linewidth=1, edgecolors='black', zorder=1, alpha=0.75)

        # Fit and plot linear model to data

        fit_x_vals = np.linspace(0, 9, 100)
        line_slope, line_intercept, line_r_value, line_p_value, line_std_err = \
            linregress(df['environment'], df['area'])
        ax.plot(fit_x_vals, line_intercept + line_slope * fit_x_vals, color='black', linestyle=':', zorder=-1)

        # Plot linear model r value
        ax.text(0.05, 0.95, '$\it{r}$' + ' = {:.{prec}f}'.format(line_r_value, prec=3),
                ha='left', va='top', transform=ax.transAxes)

        # Plot place field density

        ax_inset_height = (line_intercept + line_slope * environments[-1] * 0.6) / ax.get_ylim()[1] * 100 * 0.60
        ax_inset = inset_axes(ax, width='40%', height='{:.0f}%'.format(ax_inset_height), loc='lower right')
        ax_inset.xaxis.tick_top()
        ax_inset.scatter(df['environment'] + np.random.uniform(-0.25, 0.25, df['environment'].size),
                         df['area per m2'], s=25, c=[colors_dict[x] for x in df['animal']],
                         linewidth=1, edgecolors='black', zorder=1, alpha=0.75)

        # Adjust axes parameters

        ax.set_ylim((0, ax.get_ylim()[1]))
        ax.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax.set_xticklabels(experiment_ids_with_areas_ticks['ticklabels'])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_xlabel('environment, size (m$^2$)')
        ax.set_ylabel('proportion of field area in environment')

        ax_inset.set_ylim((0, ax_inset.get_ylim()[1] * 2))
        ax_inset.set_xlim((0, ax_inset.get_xlim()[1]))
        ax_inset.set_xticks(experiment_ids_with_areas_ticks['ticks'])
        ax_inset.set_xlabel('size (m$^2$)')
        ax_inset.set_ylabel('proportional\narea / m$^2$')
        plt.setp(ax_inset.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
        ax_inset.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_inset.xaxis.set_label_position('top')
        ax_inset.xaxis.set_ticks_position('top')
        ax_inset.yaxis.labelpad = 0

        # Fit and plot inset linear model to data

        inset_line_slope, inset_line_intercept, inset_line_r_value, inset_line_p_value, inset_line_std_err = \
            linregress(df['environment'], df['area per m2'])
        ax_inset.plot(fit_x_vals, inset_line_intercept + inset_line_slope * fit_x_vals,
                      color='black', linestyle=':', zorder=-1)

        # Compute stats

        kruskal_h_value, kruskal_pvalue = \
            kruskal(*[df[df['environment'] == group]['area'] for group in environments])

        kruskal_h_value_density, kruskal_pvalue_density = \
            kruskal(*[df[df['environment'] == group]['area per m2'] for group in environments])

        df_sorted = df.sort_values('animal')

        friedman_chisq_value, friedman_pvalue = \
            friedmanchisquare(*[df_sorted[df_sorted['environment'] == group]['area']
                                for group in environments])

        friedman_chisq_value_density, friedman_pvalue_density = \
            friedmanchisquare(*[df_sorted[df_sorted['environment'] == group]['area per m2']
                                for group in environments])

        # Plot stats to stat_ax

        stat_ax.set_title('Place field formation')
        table_cell_text = [['Field areas', 'H-value', 'p-value'],
                           ['Kruskal-Wallis test',
                            '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
                           ['', '', ''],
                           ['Field areas per m2', 'H-value', 'p-value'],
                           ['Kruskal-Wallis test',
                            '{:.2e}'.format(kruskal_h_value_density), '{:.2e}'.format(kruskal_pvalue_density)],
                           ['', '', ''],
                           ['Field areas', 'chi-square statistic', 'p-value'],
                           ['Friedman test',
                            '{:.2e}'.format(friedman_chisq_value), '{:.2e}'.format(friedman_pvalue)],
                           ['', '', ''],
                           ['Field areas per m2', 'chi-square statistic', 'p-value'],
                           ['Friedman test',
                            '{:.2e}'.format(friedman_chisq_value_density), '{:.2e}'.format(friedman_pvalue_density)],
                           ['', '', ''],
                           ['fitted linear model', 'parameters', ''],
                           ['', 'line_slope', '{:.3f}'.format(line_slope)],
                           ['', 'line_intercept', '{:.3f}'.format(line_intercept)],
                           ['', 'line_r_value', '{:.3f}'.format(line_r_value)],
                           ['', 'line_p_value', '{:.3e}'.format(line_p_value)],
                           ['', 'line_std_err', '{:.3f}'.format(line_std_err)],
                           ['', '', ''],
                           ['Inset:', '', ''],
                           ['fitted linear model', 'parameters', ''],
                           ['', 'line_slope', '{:.3f}'.format(inset_line_slope)],
                           ['', 'line_intercept', '{:.3f}'.format(inset_line_intercept)],
                           ['', 'line_r_value', '{:.3f}'.format(inset_line_r_value)],
                           ['', 'line_p_value', '{:.3e}'.format(inset_line_p_value)],
                           ['', 'line_std_err', '{:.3f}'.format(inset_line_std_err)]]
        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def make_figure(df_units, df_fields):

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.subplots_adjust(left=0.15, bottom=0.2, right=0.993, top=0.95, hspace=0.45)

        stat_fig, stat_ax = plt.subplots(1, 1, figsize=(10, 10))
        plt.tight_layout(pad=1)
        stat_ax.set_xticks([], [])
        stat_ax.set_yticks([], [])

        FieldAreaDistribution.plot_field_area_distribution(df_units, df_fields, ax, stat_ax)

        return fig, stat_fig

    @staticmethod
    def write(fpath, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'FieldAreaDistribution'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FieldAreaDistribution.make_figure(df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FiringRateChange(object):

    @staticmethod
    def plot_change_in_runs_in_one_direction_to_wall(df, value, filtering_rule, ax, stat_ax):

        if filtering_rule == 'alonglongwall':
            direction_rule = 'orthogonal_to_short_wall'
            ylabel = 'population vector change\n' + r'in runs $\bf\bot$ to short wall (Hz/cm)'
        elif filtering_rule == 'alongshortwall':
            direction_rule = 'parallel_to_short_wall'
            ylabel = 'population vector change\n' + r'in runs $\bf\parallel$ to short wall (Hz/cm)'
        else:
            raise ValueError('Unknown orientation: {}'.format(filtering_rule))

        colors_dict = {animal: color for animal, color in zip(sorted(df['animal'].unique()), sns_animal_colors)}

        distance_bins = sorted(np.unique(df['distance to short wall (cm)']))
        bin_edges = np.arange(0, df['distance to short wall (cm)'].max() + FieldDensity.bin_size, FieldDensity.bin_size)
        BinnedByDistancePlots.plot_value_binned_by_distance_as_scatter_for_one_environment(
            df, 'distance to short wall (cm)', bin_edges,
            value, 'D', 0.1 * np.max(np.diff(distance_bins)), ax, stat_ax, colors_dict,
            filter_to_middle_third=False,
            filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=direction_rule
        )

        ax.set_xlabel('distance to short wall (cm)')
        ax.set_ylabel(ylabel)

    @staticmethod
    def plot_stats_comparing_orthogonal_and_parallel_runs(df_orthogonal, df_parallel, binned_measure, value, stat_ax):
        agg_columns = ['animal', 'environment', binned_measure]
        df_orthogonal = df_orthogonal[['animal', 'environment', binned_measure, value]]
        df_orthogonal = df_orthogonal.groupby(agg_columns).mean().reset_index()
        df_parallel = df_parallel[['animal', 'environment', binned_measure, value]]
        df_parallel = df_parallel.groupby(agg_columns).mean().reset_index()

        bins = sorted(df_orthogonal[binned_measure].unique())

        diff = {}
        for name, df in zip(('orthogonal', 'parallel'), (df_orthogonal, df_parallel)):
            df_first = df.loc[df[binned_measure] == bins[0], ['animal', value]].rename(columns={value: '1st'})
            df_second = df.loc[df[binned_measure] == bins[1], ['animal', value]].rename(columns={value: '2nd'})
            df = df_first.merge(df_second, on='animal', how='inner')
            diff[name] = df['2nd'] - df['1st']

        diff_statistic, diff_pvalue = mannwhitneyu(diff['orthogonal'], diff['parallel'])

        bin_stats = []
        for bin in bins:
            bin_stats.append(
                mannwhitneyu(df_orthogonal.loc[df_orthogonal[binned_measure] == bin, value],
                             df_parallel.loc[df_parallel[binned_measure] == bin, value])
            )

        table_cell_text = [['diff', 'stat', 'p-value'],
                           ['', '{:.4f}'.format(diff_statistic), '{:.4f}'.format(diff_pvalue)],
                           ['', '', ''],
                           ['diff name', 'diff value', '']]
        for name, diff_value in diff.items():
            table_cell_text.append([name, '{:.3f}'.format(np.mean(diff[name])), ''])

        table_cell_text.append(['', '', ''])
        table_cell_text.append(['bin', 'stat', 'pvalue'])
        for bin, stats in zip(bins, bin_stats):
            table_cell_text.append([str(bin), '{:.4f}'.format(stats[0]), '{:.4f}'.format(stats[1])])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def add_theta_frequency_column(all_recordings, df, df_units):

        df['theta_frequency'] = np.nan
        for animal in df['animal'].unique():
            for experiment_id in df['experiment_id'].unique():

                animal_and_experiment_done = False

                for recordings in all_recordings:
                    for recording in recordings:

                        if animal_and_experiment_done:
                            continue

                        if recording.info['animal'] != animal or recording.info['experiment_id'] != experiment_id:
                            continue

                        place_cell_count_per_hemisphere = \
                            df_units.loc[
                                (df_units['animal'] == animal) & (df_units['category'] == 'place_cell')
                                , 'channel_group'
                            ].value_counts()

                        theta_frequency = FrequencyBandFrequency(recording, 'theta_frequency')

                        tetrode_index = \
                            theta_frequency.data['channel_labels'].index(place_cell_count_per_hemisphere.idxmax())

                        idx = (df['animal'] == animal) & (df['experiment_id'] == experiment_id)
                        df.loc[idx, 'theta_frequency'] = \
                            theta_frequency.get_values_interpolated_to_timestamps(
                                df.loc[idx, 'timestamp'].values
                            )[:, tetrode_index]

                        animal_and_experiment_done = True

    @staticmethod
    def add_smoothed_speed_column(all_recordings, df):

        df['running_speed'] = np.nan
        for animal in df['animal'].unique():
            for experiment_id in df['experiment_id'].unique():

                animal_and_experiment_done = False

                for recordings in all_recordings:
                    for recording in recordings:

                        if animal_and_experiment_done:
                            continue

                        if recording.info['animal'] != animal or recording.info['experiment_id'] != experiment_id:
                            continue

                        idx = (df['animal'] == animal) & (df['experiment_id'] == experiment_id)

                        speed = recording.get_smoothed_speed(Params.xy_masking['speed_smoothing_window'])

                        df.loc[idx, 'running_speed'] = \
                            np.interp(df.loc[idx, 'timestamp'].values, recording.position['timestamps'], speed)

                        animal_and_experiment_done = True

    @staticmethod
    def make_figure(fpath, all_recordings, df_units, verbose=False):

        fig, axs = plt.subplots(1, 2, figsize=(7, 4))
        plt.subplots_adjust(left=0.13, bottom=0.16, right=0.95, top=0.8, wspace=0.65)

        stat_fig, stat_axs = plt.subplots(1, 3, figsize=(30, 20))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs.flatten():
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        binned_measure = 'distance to short wall (cm)'
        value = 'rate change\n(euclidean)'

        df = PopulationVectorChangeRate.get_dataframe(all_recordings)
        df.dropna(inplace=True)
        df = df.loc[df['environment'] == 'D'].copy()

        FiringRateChange.add_theta_frequency_column(all_recordings, df, df_units)
        FiringRateChange.add_smoothed_speed_column(all_recordings, df)

        # Write DataFrame to disk for use in other analyses
        population_vector_change_file_path = construct_df_population_vector_change_file_path(fpath)
        df.to_pickle(population_vector_change_file_path)
        print('Population vector change values written to {}'.format(population_vector_change_file_path))

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure,
                                                      FieldDensity.bin_size)
        df.drop(index=df.loc[df[binned_measure] > FieldDensity.max_bin_center].index, inplace=True)

        df_alonglongwall = filter_dataframe_by_direction(
            df.copy(deep=True), 'alonglongwall', section_width='quadrants'
        )
        df_alongshortwall = filter_dataframe_by_direction(
            df.copy(deep=True), 'alongshortwall', section_width='quadrants'
        )

        for df_tmp, filtering_rule, ax, stat_ax in zip((df_alonglongwall, df_alongshortwall),
                                                       ('alonglongwall', 'alongshortwall'),
                                                       axs, stat_axs[:2]):

            df_tmp = df_tmp[['animal', 'environment', binned_measure, value]]
            agg_columns = ['animal', 'environment', binned_measure]
            df_tmp = df_tmp.groupby(agg_columns).mean().reset_index()

            FiringRateChange.plot_change_in_runs_in_one_direction_to_wall(df_tmp, value, filtering_rule, ax, stat_ax)

        ylim = get_max_ylim(axs)
        axs[0].set_ylim(ylim)
        axs[1].set_ylim(ylim)

        FiringRateChange.plot_stats_comparing_orthogonal_and_parallel_runs(
            df_alonglongwall, df_alongshortwall, binned_measure, value, stat_axs[2]
        )

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, df_units, prefix='', verbose=True):

        figure_name = prefix + 'FiringRateChange'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FiringRateChange.make_figure(fpath, all_recordings, df_units, verbose=verbose)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FiringRateChangeAll(object):
    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def plot_field_change_by_distance_to_wall(df, value, binned_measure, direction_rule,
                                              ylabel, ax, stat_ax):

        df = df[['animal', 'environment', binned_measure, value]].dropna()
        df = df.groupby(['animal', 'environment', binned_measure]).mean().reset_index()

        bin_edges = np.arange(0, df[binned_measure].max() + FieldDensity.bin_size, FieldDensity.bin_size)

        return BinnedByDistancePlots.plot_value_binned_by_measure_as_scatter_by_environment(
            df, ax, stat_ax, binned_measure, bin_edges, value,
            'distance to short wall (cm)', ylabel,
            filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
            orientation_rule=None, direction_rule=direction_rule,
            plot_first_bin_comparison_between_environments=False
        )

    @staticmethod
    def make_figure(all_recordings):

        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        plt.subplots_adjust(left=0.1, bottom=0.16, right=0.98, top=0.86, wspace=0.4)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(40, 15))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs:
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        binned_measure = 'distance to short wall (cm)'
        value = 'rate change\n(euclidean)'

        df = PopulationVectorChangeRate.get_dataframe(all_recordings)

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure,
                                                      FieldDensity.bin_size)
        df.drop(index=df.loc[df[binned_measure] > FieldDensity.max_bin_center].index, inplace=True)

        df_alonglongwall = filter_dataframe_by_direction(
            df.copy(deep=True), 'alonglongwall', section_width='quadrants'
        )
        df_alongshortwall = filter_dataframe_by_direction(
            df.copy(deep=True), 'alongshortwall', section_width='quadrants'
        )

        ylabels = ('population vector change\n' + r'in runs $\bf\bot$ to short wall (Hz/cm)',
                   'population vector change\n' + r'in runs $\bf\parallel$ to short wall (Hz/cm)')
        direction_rules = ('orthogonal_to_short_wall', 'parallel_to_short_wall')

        all_axs = []
        for df_tmp, direction_rule, ylabel, ax, stat_ax in zip((df_alonglongwall, df_alongshortwall),
                                                               direction_rules,
                                                               ylabels, axs, stat_axs):

            all_axs += FiringRateChangeAll.plot_field_change_by_distance_to_wall(
                df_tmp, value, binned_measure, direction_rule, ylabel, ax, stat_ax
            )

        ylim = get_max_ylim(all_axs)
        for ax in all_axs:
            ax.set_ylim(ylim)

        return fig, stat_fig

    @staticmethod
    def write(fpath, all_recordings, prefix='', verbose=True):

        figure_name = prefix + 'FiringRateChangeAll'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FiringRateChangeAll.make_figure(all_recordings)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.svg'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class FiringRateChangeAndTheta(object):

    @staticmethod
    def binned_values(values, quantiles=10):
        return np.array(list(map(lambda c: float(c.mid.mean()), pd.qcut(values, quantiles))))

    @staticmethod
    def plot_single_relation(df, x_axis_column, y_axis_column, covar_column, ax, stat_ax):

        df = df.copy(deep=True)

        df[x_axis_column] = FiringRateChangeAndTheta.binned_values(df[x_axis_column])
        df.loc[df[x_axis_column] == df[x_axis_column].min(), x_axis_column] = np.nan
        df.loc[df[x_axis_column] == df[x_axis_column].max(), x_axis_column] = np.nan

        df = df.dropna()

        dfg = df.groupby(['animal', x_axis_column])[[y_axis_column, covar_column]].mean().reset_index()

        sns.scatterplot(
            data=dfg,
            x=x_axis_column,
            y=y_axis_column,
            hue='animal',
            ax=ax)

        pcorr_stats = partial_corr(
            data=dfg,
            x=x_axis_column,
            y=y_axis_column,
            covar=covar_column
        )

        ax.set_title('r = {:.3f} | p = {:e}'.format(pcorr_stats.loc['pearson', 'r'],
                                                    pcorr_stats.loc['pearson', 'p-val']))

        table_cell_text = []
        table_cell_text.append(['x', x_axis_column])
        table_cell_text.append(['', ''])
        table_cell_text.append(['', ''])
        table_cell_text.append(['y', y_axis_column])
        table_cell_text.append(['', ''])
        table_cell_text.append(['', ''])
        table_cell_text.append(['covar', covar_column])
        table_cell_text.append(['', ''])
        table_cell_text.append(['', ''])
        for stat_column in pcorr_stats.columns:
            table_cell_text.append([stat_column, str(pcorr_stats.loc['pearson', stat_column])])
            table_cell_text.append(['', ''])

        stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')

    @staticmethod
    def normalise_values_per_animal(df, columns):
        for column in columns:
            for animal in df['animal'].unique():
                idx = df['animal'] == animal
                df.loc[idx, column] = zscore(df.loc[idx, column])

    @staticmethod
    def plot(df, axs, stat_axs):

        df.rename(columns={'rate change\n(euclidean)': 'population activity change rate (z score)',
                           'theta_frequency': 'theta frequency (z score)',
                           'running_speed': 'running speed (z score)'},
                  inplace=True)

        FiringRateChangeAndTheta.normalise_values_per_animal(df, ['theta frequency (z score)', 'running speed (z score)',
                                                                  'population activity change rate (z score)'])

        FiringRateChangeAndTheta.plot_single_relation(df, 'theta frequency (z score)', 'running speed (z score)',
                                                      'population activity change rate (z score)',
                                                      axs[0], stat_axs[0])

        FiringRateChangeAndTheta.plot_single_relation(df, 'population activity change rate (z score)',
                                                      'theta frequency (z score)', 'running speed (z score)',
                                                      axs[1], stat_axs[1])

    @staticmethod
    def make_figure(fpath):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.98, top=0.92, wspace=0.4, hspace=0.4)

        stat_fig, stat_axs = plt.subplots(1, 2, figsize=(14, 8))
        plt.tight_layout(pad=1.5)
        for stat_ax in stat_axs.flatten():
            stat_ax.set_xticks([], [])
            stat_ax.set_yticks([], [])

        FiringRateChangeAndTheta.plot(
            pd.read_pickle(construct_df_population_vector_change_file_path(fpath)), axs, stat_axs
        )

        return fig, stat_fig

    @staticmethod
    def write(fpath, prefix='', verbose=True):

        figure_name = prefix + 'FiringRateChangeAndTheta'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig, stat_fig = FiringRateChangeAndTheta.make_figure(fpath)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        stat_fig.savefig(os.path.join(paper_figures_path(fpath), '{}_stats.png'.format(figure_name)))
        plt.close(fig)
        plt.close(stat_fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


class StabilityAndLastWallPlots(object):

    wall_touch_threshold = 20
    wall_markers = ['N', 'E', 'S', 'W']

    @staticmethod
    def get_df_with_place_cell_fields_in_middle(df_units, df_fields):
        df = df_fields.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                      how='left', on=['animal', 'animal_unit'])
        df = df[(df['category'] == 'place_cell') & (df['experiment_id'] == 'exp_scales_d')]
        return df.loc[(df['peak_nearest_wall'] > 50) & (df['peak_nearest_wall'] < 100)].copy()

    @staticmethod
    def position_sample_wall_indices(position_xy):

        east_wall = 350 - StabilityAndLastWallPlots.wall_touch_threshold
        south_wall = 250 - StabilityAndLastWallPlots.wall_touch_threshold

        last_wall = 'X'
        position_sample_wall_indices = []
        for x, y in position_xy:

            if x < StabilityAndLastWallPlots.wall_touch_threshold:
                last_wall = 'W'
            elif x > east_wall:
                last_wall = 'E'
            elif y < 20:
                last_wall = 'N'
            elif y > south_wall:
                last_wall = 'S'

            position_sample_wall_indices.append(last_wall)

        return np.array(position_sample_wall_indices)

    @staticmethod
    def plot_spike_rate_plot(ax, recording, df, legend=False):
        ax.plot(*recording.position['xy'].T, 'k', linewidth=0.2, zorder=-1)

        sns.scatterplot(data=df, x='x', y='y', hue='wall', ax=ax,
                        s=10, palette='colorblind', legend=legend,
                        hue_order=StabilityAndLastWallPlots.wall_markers, zorder=1)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(spatial_windows['exp_scales_d'][:2])
        ax.set_ylim(spatial_windows['exp_scales_d'][2:][::-1])
        ax.set_xlabel('')
        ax.set_ylabel('')

        if legend:
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=4)

    @staticmethod
    def plot_spike_plots(axs, recording, spike_timestamps, position_sample_indices,
                         unit_position_sample_wall_indices, legend=False):

        df = pd.DataFrame({
            'x': recording.position['xy'][position_sample_indices, 0],
            'y': recording.position['xy'][position_sample_indices, 1],
            'timestamp': spike_timestamps,
            'wall': unit_position_sample_wall_indices
        })

        recording_midpoint = (recording.info['last_timestamp'] - recording.info['first_timestamp']) / 2.

        StabilityAndLastWallPlots.plot_spike_rate_plot(
            axs[0], recording, df.loc[df['timestamp'] < recording_midpoint], legend=legend
        )
        StabilityAndLastWallPlots.plot_spike_rate_plot(
            axs[1], recording, df.loc[df['timestamp'] >= recording_midpoint], legend=False
        )

    @staticmethod
    def get_spikes_and_position(recording, unit, position_sample_wall_indices):

        spike_timestamps = unit['timestamps']
        position_sample_indices = convert_spike_times_to_sample_indices(spike_timestamps,
                                                                        recording.position['sampling_rate'])
        unit_position_sample_wall_indices = position_sample_wall_indices[position_sample_indices]

        # Filter by speed
        idx_passed_filter = recording.position['analysis']['ratemap_speed_mask'][position_sample_indices]
        # Filter by position not near boundary
        spike_xy = recording.position['xy'][position_sample_indices, :]
        east_wall = 350 - StabilityAndLastWallPlots.wall_touch_threshold
        south_wall = 250 - StabilityAndLastWallPlots.wall_touch_threshold
        idx_passed_filter = (
                idx_passed_filter
                & (spike_xy[:, 0] > StabilityAndLastWallPlots.wall_touch_threshold)
                & (spike_xy[:, 1] > StabilityAndLastWallPlots.wall_touch_threshold)
                & (spike_xy[:, 0] < east_wall)
                & (spike_xy[:, 1] < south_wall)
        )
        # Filter by valid wall
        idx_passed_filter = idx_passed_filter & (unit_position_sample_wall_indices != 'X')
        # Apply filter
        spike_timestamps = spike_timestamps[idx_passed_filter]
        position_sample_indices = position_sample_indices[idx_passed_filter]
        unit_position_sample_wall_indices = unit_position_sample_wall_indices[idx_passed_filter]

        return spike_timestamps, position_sample_indices, unit_position_sample_wall_indices

    @staticmethod
    def plot_wall_touch_ratemap(ax, recording, spike_timestamps, position_sample_indices,
                                unit_position_sample_wall_indices):

        spatial_window = spatial_windows['exp_scales_d']

        ratemaps = []
        ratemap_shape = None
        peak_rate = 0
        for wall_marker in StabilityAndLastWallPlots.wall_markers:

            idx_wall = unit_position_sample_wall_indices == wall_marker

            if not np.any(idx_wall):
                ratemaps.append(None)
                continue

            spatial_ratemap = SpatialRatemap(
                recording.position['xy'],
                spike_timestamps[idx_wall],
                recording.position['sampling_rate'],
                Params.spatial_ratemap['bin_size'],
                spatial_window=spatial_window,
                spike_xy_ind=position_sample_indices[idx_wall],
                xy_mask=recording.position['analysis']['ratemap_speed_mask']
            )
            ratemap = spatial_ratemap.spike_rates_smoothed(
                n_bins=Params.spatial_ratemap['n_smoothing_bins'],
                method=Params.spatial_ratemap['smoothing_method']
            )
            ratemaps.append(ratemap)
            ratemap_shape = ratemap.shape
            peak_rate = max(peak_rate, np.nanmax(ratemap))

        aggregate_ratemap = np.zeros((*ratemap_shape, 3))
        if not (ratemaps[0] is None):
            aggregate_ratemap[:, :, 0] = ratemaps[0] / peak_rate
        if not (ratemaps[1] is None):
            aggregate_ratemap[:, :, 1] = ratemaps[1] / peak_rate
        if not (ratemaps[2] is None):
            aggregate_ratemap[:, :, 2] = ratemaps[2] / peak_rate

        aggregate_ratemap = aggregate_ratemap / 2
        if not (ratemaps[3] is None):
            aggregate_ratemap = aggregate_ratemap + np.stack([(ratemaps[3] / peak_rate) / 2] * 3, axis=2)

        extent = (spatial_window[0], spatial_window[1],
                  spatial_window[3], spatial_window[2])

        ax.imshow(aggregate_ratemap, extent=extent)

    @staticmethod
    def plot_both_wall_touch_ratemaps(axs, recording, spike_timestamps,
                                      position_sample_indices, unit_position_sample_wall_indices):

        recording_midpoint = (recording.info['last_timestamp'] - recording.info['first_timestamp']) / 2.

        idx_1st_half = spike_timestamps < recording_midpoint
        idx_2nd_half = spike_timestamps >= recording_midpoint

        for ax, spike_mask in zip(axs, (idx_1st_half, idx_2nd_half)):
            StabilityAndLastWallPlots.plot_wall_touch_ratemap(
                ax, recording, spike_timestamps[spike_mask], position_sample_indices[spike_mask],
                unit_position_sample_wall_indices[spike_mask]
            )

    @staticmethod
    def plot(all_recordings, df, axs):

        animal_recordings = {}
        animal_recording_indices = {}
        position_sample_wall_indices = {}
        for recordings in all_recordings:
            for i_recording, recording in enumerate(recordings):
                if recording.info['experiment_id'] == 'exp_scales_d':
                    animal_recordings[recording.info['animal']] = recordings
                    animal_recording_indices[recording.info['animal']] = i_recording
                    position_sample_wall_indices[recording.info['animal']] = \
                        StabilityAndLastWallPlots.position_sample_wall_indices(recording.position['xy'])

        legend = True
        for i, unit_index in enumerate(df['unit'].unique()):

            df_unit_fields = df.loc[df['unit'] == unit_index]
            field_row = df_unit_fields.iloc[0]
            animal = field_row['animal']
            animal_unit_index = field_row['animal_unit']

            recordings = animal_recordings[animal]
            recording = recordings[animal_recording_indices[animal]]
            unit = recordings.units[animal_unit_index][animal_recording_indices[animal]]

            spike_timestamps, position_sample_indices, unit_position_sample_wall_indices = \
                StabilityAndLastWallPlots.get_spikes_and_position(recording, unit, position_sample_wall_indices[animal])

            table_cell_text = [['animal', animal],
                               ['', ''],
                               ['animal_unit', str(animal_unit_index)],
                               ['', ''],
                               ['animal_fields', ','.join(map(str, df_unit_fields['animal_field']))]]

            axs[i, 0].table(cellText=table_cell_text, cellLoc='left', loc='center left', edges='open')
            axs[i, 0].axis('off')

            ratemap_half_a = unit['analysis']['spatial_ratemaps']['spike_rates_halves']['first']
            ratemap_half_b = unit['analysis']['spatial_ratemaps']['spike_rates_halves']['second']

            SpatialRatemap.plot(ratemap_half_a, spatial_windows['exp_scales_d'], ax=axs[i, 1],
                                colorbar=True, cmap='jet')
            SpatialRatemap.plot(ratemap_half_b, spatial_windows['exp_scales_d'], ax=axs[i, 2],
                                colorbar=True, cmap='jet')

            StabilityAndLastWallPlots.plot_spike_plots(
                axs[i, 3:5], recording,
                spike_timestamps, position_sample_indices,
                unit_position_sample_wall_indices,
                legend=legend
            )
            if legend:
                legend = False

            StabilityAndLastWallPlots.plot_both_wall_touch_ratemaps(
                axs[i, 5:], recording, spike_timestamps, position_sample_indices, unit_position_sample_wall_indices
            )

            contours = [compute_field_contour(recordings[0].analysis['fields'][i]['ratemap'])
                        for i in df_unit_fields['animal_field']]
            contours_spatial_window = \
                list(spatial_windows['exp_scales_d'][:2]) + list(spatial_windows['exp_scales_d'][2:])[::-1]
            for contour in contours:
                for ax in axs[i, 1:]:
                    SpatialRatemap.plot_contours(
                        contour, ax, color='green',
                        ratemap_shape=ratemap_half_a.shape,
                        spatial_window=contours_spatial_window
                    )

    @staticmethod
    def make_figure(all_recordings, df_units, df_fields):

        df = StabilityAndLastWallPlots.get_df_with_place_cell_fields_in_middle(df_units, df_fields)
        number_of_units = df['unit'].unique().size

        fig, axs = plt.subplots(number_of_units, 7, figsize=(30, 3 * number_of_units))
        plt.subplots_adjust(left=0.025, bottom=0.01, right=0.99, top=0.95, wspace=0.4, hspace=0.01 / number_of_units)

        StabilityAndLastWallPlots.plot(
            all_recordings, df, axs
        )

        return fig

    @staticmethod
    def write(fpath, all_recordings, df_units, df_fields, prefix='', verbose=True):

        figure_name = prefix + 'StabilityAndLastWallPlots'

        if verbose:
            print('Writing Figure {}'.format(figure_name))

        sns.set(context='paper', style='ticks', palette='muted', font_scale=seaborn_font_scale)

        fig = StabilityAndLastWallPlots.make_figure(all_recordings, df_units, df_fields)
        fig.savefig(os.path.join(paper_figures_path(fpath), '{}.png'.format(figure_name)))
        plt.close(fig)

        if verbose:
            print('Writing Figure {} Done.'.format(figure_name))


def print_field_count_per_cell_correlation_with_clustering_quality(df_units, df_fields):

    df = df_fields.loc[df_fields['experiment_id'] == 'exp_scales_d',
                       ['animal', 'animal_unit', 'area']].copy(deep=True)
    df = df.merge(df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                  how='left', on=['animal', 'animal_unit'])
    df = df.loc[df['category'] == 'place_cell', ['animal', 'animal_unit', 'area']]  # Only keep place cell fields
    df['count'] = 1
    df = df.groupby(['animal', 'animal_unit'])['count', 'area'].sum().reset_index()
    df['mean_area'] = df['area'] / df['count']

    df = df.merge(df_units[['animal', 'animal_unit', 'isolation_distance', 'l_ratio']],
                  on=['animal', 'animal_unit'],
                  how='left')
    df = df.dropna()

    for correlate in ('count', 'mean_area'):

        print()
        print('Comparing {} correlation with clustering quality measures'.format(correlate))

        for measure in ('isolation_distance', 'l_ratio'):

            print()
            print('Clustering quality measure: {}'.format(measure))

            print('Across animals correlation of {} to field {}: r={:.3f} p={:.6f}'.format(measure, correlate,
                                                                                           *pearsonr(df[measure],
                                                                                                     df[correlate])))
            for animal in df['animal'].unique():
                idx = df['animal'] == animal
                print(
                    'Animal {} correlation of {} to field {}: r={:.3f} p={:.6f} | total units = {} | {} per unit {:.2f}'.format(
                        animal, measure, correlate, *pearsonr(df.loc[idx, measure], df.loc[idx, correlate]), np.sum(idx),
                        correlate, np.mean(df.loc[idx, correlate])))


def load_data_preprocessed_if_available(fpath, recompute=False, verbose=False):

    # This ensures all possible pre-processing is completed before loading data
    # If pre-processing has not been run on the data yet, this step is very slow
    # and requires large amounts of CPU memory to run.

    # Preferably, the file barrylab_ephys_analysis/scripts/exp_scales/paper_preprocess.py
    # would be run as a script with the same input as this file on a powerful machine.
    # The script will compute and save the computationally expensive parts of the analysis
    # to the NWB files. If this is done before launching barrylab_ephys_analysis/scripts/exp_scales/paper_figures.py
    # then the following line will purely do some verification that computations have completed.
    preprocess_and_save_all_animals(fpath, recompute=recompute, verbose=verbose)
    print('Preprocessing complete for all animals.')

    # Load data from all animals into memory
    all_recordings = load.load_recordings_of_all_animals(
        fpath, Params.animal_ids, continuous_data_type=None, no_waveforms=True,
        clustering_name=Params.clustering_name, verbose=verbose
    )
    # Load pre-processing results to memory
    for recordings in all_recordings:
        recordings.load_analysis(ignore=('waveform_properties',))

    return all_recordings


def get_full_df_units(all_recordings):
    return pd.concat([recordings.df_units for recordings in all_recordings], ignore_index=True)


def get_full_df_fields(all_recordings):
    return pd.concat([recordings.df_fields for recordings in all_recordings], ignore_index=True)


def link_df_units_and_df_fields_with_common_unit(df_units, df_fields):

    df_units_tmp = df_units[['animal', 'animal_unit']].copy()
    df_units_tmp['unit'] = list(range(df_units_tmp.shape[0]))

    df_merge = pd.merge(left=df_units_tmp, right=df_fields[['animal', 'animal_unit']],
                        on=['animal', 'animal_unit'])

    df_fields.insert(0, 'unit', df_merge['unit'])


def main(fpath):

    # all_recordings = load_data_preprocessed_if_available(fpath, recompute=False, verbose=True)
    #
    # # Rename experiment name in  last recording so it would not have the same as the first
    # for recordings in all_recordings:
    #     snippets.rename_last_recording_a2(recordings)
    #
    # for recordings in all_recordings:
    #     create_df_fields_for_recordings(recordings)
    #
    # for recordings in all_recordings:
    #     create_unit_data_frames_for_recordings(recordings)
    #
    # df_fields = get_full_df_fields(all_recordings)
    # df_units = get_full_df_units(all_recordings)
    # link_df_units_and_df_fields_with_common_unit(df_units, df_fields)
    #
    # df_fields.to_pickle(os.path.join(fpath, Params.analysis_path, 'df_fields.p'))
    # df_units.to_pickle(os.path.join(fpath, Params.analysis_path, 'df_units.p'))
    #
    # with open(os.path.join(fpath, 'Analysis', 'all_recordings.p'), 'wb') as pfile:
    #     pickle.dump(all_recordings, pfile)

    # Use this instead if data has already been loaded once
    with open(os.path.join(fpath, 'Analysis', 'all_recordings.p'), 'rb') as pfile:
        all_recordings = pickle.load(pfile)

    df_units = pd.read_pickle(os.path.join(fpath, 'Analysis', 'df_units.p'))
    df_fields = pd.read_pickle(os.path.join(fpath, 'Analysis', 'df_fields.p'))

    # Compute and write figures

    # ExampleUnit.write(fpath, all_recordings, df_units, prefix='Figure_1_')
    # FieldDetectionMethod.write(fpath, all_recordings, df_units, prefix='Figure_1_sup_2_')
    # IntraTrialCorrelations.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_1_sup_3_')
    # PlaceCellAndFieldCounts.write(fpath, df_units, df_fields, prefix='Figure_2AB_')
    # FieldsPerCellAcrossEnvironmentsSimple.write(fpath, df_units, df_fields, prefix='Figure_2C_')
    # Remapping.write(fpath, all_recordings, prefix='Figure_2_sup_1_')
    # environment_field_density_model_parameters = \
    #     FieldsDetectedAcrossEnvironments.write(fpath, df_units, df_fields, prefix='Figure_2E_')
    # ConservationOfFieldFormationPropensity.write(fpath, df_units, df_fields,
    #                                              environment_field_density_model_parameters, prefix='Figure_2_sup_2_')
    # gamma_model_fit = \
    #     FieldsPerCellAcrossEnvironments.write(fpath, df_units, df_fields, environment_field_density_model_parameters,
    #                                           prefix='Figure_2_sup_3_')
    # PlaceCellsDetectedAcrossEnvironments.write(fpath, df_units, df_fields,
    #                                            environment_field_density_model_parameters, gamma_model_fit,
    #                                            prefix='Figure_2D_')
    # FieldDensity.write(fpath, df_units, df_fields, prefix='Figure_3A_')
    # FieldSize.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_3B_')
    # FieldWidth.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_3CD_')
    # AverageActivity.write(fpath, all_recordings, prefix='Figure_4AB_')
    # FiringRateDistribution.write(fpath, all_recordings, prefix='Figure_4C_')
    # FieldAreaDistribution.write(fpath, df_units, df_fields, prefix='Figure_4D_')
    # FieldDensityByDwell.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_3_sup_1_')
    # FieldWidthAll.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_3_sup_2_')
    # AverageActivityAll.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_4_sup_1_')
    # InterneuronMeanRate.write(fpath, all_recordings, prefix='Figure_4_sup_2_')
    # FiringRateChange.write(fpath, all_recordings, df_units, prefix='Figure_5AB_')
    # FiringRateChangeAll.write(fpath, all_recordings, prefix='Figure_5_sup_1_')
    # FiringRateChangeAndTheta.write(fpath, prefix='Figure_R1_')
    StabilityAndLastWallPlots.write(fpath, all_recordings, df_units, df_fields, prefix='Figure_R2_')

    # print_field_count_per_cell_correlation_with_clustering_quality(df_units, df_fields)


if __name__ == '__main__':
    main(sys.argv[1])
