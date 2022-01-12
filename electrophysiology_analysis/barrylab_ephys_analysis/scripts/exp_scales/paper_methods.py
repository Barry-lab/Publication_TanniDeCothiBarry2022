import os
import warnings
import pprint
from shutil import rmtree
import numpy as np
import textwrap
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.modeling.functional_models import Gaussian2D
from scipy.interpolate import interp1d
from scipy.stats import circmean, wilcoxon, levene
from scipy.signal import convolve2d, savgol_filter
from copy import deepcopy
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare
from itertools import combinations
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpecFromSubplotSpec
import seaborn as sns

from barrylab_ephys_analysis.blea_utils import euclidean_distance_between_rows_of_matrix
from barrylab_ephys_analysis.recording_io import Recording
from barrylab_ephys_analysis.scripts.exp_scales import snippets
from barrylab_ephys_analysis.scripts.exp_scales.params import Params
from barrylab_ephys_analysis.external.statannot import add_stat_annotation
from barrylab_ephys_analysis.external import ptitprince
from barrylab_ephys_analysis.spatial.fields import compute_field_contour
from barrylab_ephys_analysis.spatial.measures import ratemap_gradient, ratemap_fisher_information
from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap
from barrylab_ephys_analysis.spatial.similarity import spatial_correlation
from barrylab_ephys_analysis.spikes.utils import count_spikes_in_sample_bins


pvalue_thresholds = [[0.001, '***'], [0.01, '**'], [0.05, '*'], [1, None]]


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


def compute_distances_to_short_and_long_walls(df, xy):
    longwallmap = {
        'exp_scales_a': (xy[:, 0], spatial_windows['exp_scales_a'][0:2]),
        'exp_scales_b': (xy[:, 1], spatial_windows['exp_scales_b'][2:4]),
        'exp_scales_c': (xy[:, 0], spatial_windows['exp_scales_c'][0:2]),
        'exp_scales_d': (xy[:, 1], spatial_windows['exp_scales_d'][2:4])
    }
    shortwallmap = {
        'exp_scales_a': (xy[:, 1], spatial_windows['exp_scales_a'][2:4]),
        'exp_scales_b': (xy[:, 0], spatial_windows['exp_scales_b'][0:2]),
        'exp_scales_c': (xy[:, 1], spatial_windows['exp_scales_c'][2:4]),
        'exp_scales_d': (xy[:, 0], spatial_windows['exp_scales_d'][0:2])
    }

    if np.any(df['experiment_id'] == 'exp_scales_a2'):
        raise ValueError('exp_scales_a2 not handled')

    df['distance to long wall (cm)'] = np.nan
    df['distance to short wall (cm)'] = np.nan
    df['distance_to_long_wall_prop'] = np.nan
    df['distance_to_short_wall_prop'] = np.nan
    for experiment_id in longwallmap:
        idx = df['experiment_id'] == experiment_id

        position, edges = longwallmap[experiment_id]
        df.loc[idx, 'distance to long wall (cm)'] = \
            np.min(np.abs(position[idx, np.newaxis] - np.array(edges)[np.newaxis, :]), axis=1)

        short_wall_half_length = (edges[1] - edges[0]) / 2.

        position, edges = shortwallmap[experiment_id]
        df.loc[idx, 'distance to short wall (cm)'] = \
            np.min(np.abs(position[idx, np.newaxis] - np.array(edges)[np.newaxis, :]), axis=1)

        df.loc[idx, 'distance_to_short_wall_prop'] = \
            df.loc[idx, 'distance to short wall (cm)'] / short_wall_half_length
        df.loc[idx, 'distance_to_long_wall_prop'] = \
            df.loc[idx, 'distance to long wall (cm)'] / short_wall_half_length


def compute_distance_to_nearest_corner(df, xy):
    arena_sizes = {
        'exp_scales_a': (spatial_windows['exp_scales_a'][1], spatial_windows['exp_scales_a'][3]),
        'exp_scales_b': (spatial_windows['exp_scales_b'][1], spatial_windows['exp_scales_b'][3]),
        'exp_scales_c': (spatial_windows['exp_scales_c'][1], spatial_windows['exp_scales_c'][3]),
        'exp_scales_d': (spatial_windows['exp_scales_d'][1], spatial_windows['exp_scales_d'][3])
    }

    if np.any(df['experiment_id'] == 'exp_scales_a2'):
        raise ValueError('exp_scales_a2 not handled')

    df['distance_from_corner_prop'] = np.nan
    df['distance_from_corner (cm)'] = np.nan

    for experiment_id, arena_size in arena_sizes.items():
        idx = df['experiment_id'] == experiment_id
        df.loc[idx, 'distance_from_corner (cm)'] = \
            snippets.compute_distance_to_nearest_corner_for_array(xy[idx, :], arena_size)
        df.loc[idx, 'distance_from_corner_prop'] = \
            df.loc[idx, 'distance_from_corner (cm)'] / (np.min(arena_size) / 2)


def compute_distances_to_landmarks(df, xy):
    compute_distances_to_short_and_long_walls(df, xy)
    compute_distance_to_nearest_corner(df, xy)


def cut_df_rows_by_distance_to_wall(df, bin_width, environment_column='environment',
                                    distance_column='distance to wall (cm)',
                                    reset_index=True):

    match_to_exp_scales = np.any(np.array([x in experiment_id_substitutes for x in df[environment_column]]))
    if match_to_exp_scales:
        max_bin_centers = {
            'exp_scales_a': (87.5 / 2. // bin_width) * bin_width - bin_width / 2.,
            'exp_scales_b': (125 / 2. // bin_width) * bin_width - bin_width / 2.,
            'exp_scales_c': (175 / 2. // bin_width) * bin_width - bin_width / 2.,
            'exp_scales_d': (250 / 2. // bin_width) * bin_width - bin_width / 2.,
            'exp_scales_a2': (87.5 / 2. // bin_width) * bin_width - bin_width / 2.
        }
    else:
        max_bin_centers = {
            'A': (87.5 / 2. // bin_width) * bin_width - bin_width / 2.,
            'B': (125 / 2. // bin_width) * bin_width - bin_width / 2.,
            'C': (175 / 2. // bin_width) * bin_width - bin_width / 2.,
            'D': (250 / 2. // bin_width) * bin_width - bin_width / 2.,
            'A*': (87.5 / 2. // bin_width) * bin_width - bin_width / 2.,
            "A'": (87.5 / 2. // bin_width) * bin_width - bin_width / 2.
        }

    for environment_name in list(df[environment_column].unique()):

        df.drop(df[(df[environment_column] == environment_name)
                   & (np.array(df[distance_column]) > max_bin_centers[environment_name] + 0.0001)].index,
                inplace=True)

        if reset_index:
            df.reset_index(drop=True, inplace=True)


class SpatialFilteringLegend(object):

    def __init__(self):

        self._selection_maps = {}
        for experiment_id in main_experiment_ids:
            selection_map_shape = (
                np.int16(spatial_windows[experiment_id][3] / Params.spatial_ratemap['bin_size']),
                np.int16(spatial_windows[experiment_id][1] / Params.spatial_ratemap['bin_size']),
                3
            )
            self._selection_maps[experiment_id] = 0.7 * np.ones(selection_map_shape)

        dfs = []
        for experiment_id in main_experiment_ids:

            x_ind, y_ind = np.meshgrid(np.arange(self._selection_maps[experiment_id].shape[1]),
                                       np.arange(self._selection_maps[experiment_id].shape[0]))
            x_ind = x_ind.flatten()
            y_ind = y_ind.flatten()
            x_coord = (x_ind + 0.5) * Params.spatial_ratemap['bin_size']
            y_coord = (y_ind + 0.5) * Params.spatial_ratemap['bin_size']

            df = pd.DataFrame({'x_ind': x_ind, 'y_ind': y_ind, 'x_coord': x_coord, 'y_coord': y_coord})
            df['experiment_id'] = experiment_id

            self.compute_distance_to_wall(df, self._selection_maps[experiment_id].shape[:2],
                                          spatial_windows[experiment_id])

            dfs.append(df)

        df = pd.concat(dfs, axis=0, ignore_index=True)

        compute_distances_to_landmarks(df, np.stack((df['x_coord'], df['y_coord']), axis=1))

        self._dfs = {experiment_id: df[df['experiment_id'] == experiment_id]
                     for experiment_id in main_experiment_ids}

    @staticmethod
    def compute_distance_to_wall(df, map_shape, spatial_window):

        bin_distance_to_wall = snippets.SpatialMapBinCenterDistanceToWall()

        distance_to_wall = []
        for _, series in df.iterrows():
            distance_to_wall.append(
                bin_distance_to_wall.get(map_shape, spatial_window,
                                         series['x_ind'], series['y_ind'])
            )

        df['distance to wall (cm)'] = np.array(distance_to_wall)

    @staticmethod
    def set_map_values_to_green(selection_map, y_ind, x_ind, intensity=1):
        selection_map[y_ind, x_ind, :] = 0
        selection_map[y_ind, x_ind, 1] = intensity

    def location_map(self, experiment_id, distance_measure=None, distance_bin_width=None,
                     filter_to_middle_third=False, filter_to_middle_fifth=False,
                     filter_to_first_bin_from_wall=False, max_bin_center=None):

        df = self._dfs[experiment_id].copy(deep=True)
        selection_map = self._selection_maps[experiment_id].copy()

        if filter_to_middle_third:
            df = ValueByBinnedDistancePlot.filter_to_middle_third(df)

        if filter_to_middle_fifth:
            df = ValueByBinnedDistancePlot.filter_to_middle_fifth(df)

        if filter_to_first_bin_from_wall:
            df = ValueByBinnedDistancePlot.filter_to_first_bin_from_wall(df, distance_bin_width)

        if not (distance_measure is None):
            ValueByBinnedDistancePlot.bin_distance_values(df, distance_measure, distance_bin_width,
                                                          environment_column='experiment_id')
            if max_bin_center is not None:
                df.drop(index=df.loc[df[distance_measure] > max_bin_center].index, inplace=True)

            distance_bins = sorted(df[distance_measure].unique())

            for distance_bin, intensity in zip(distance_bins, np.linspace(0.4, 0.8, len(distance_bins))):
                tmp_df = df[df[distance_measure] == distance_bin]
                self.set_map_values_to_green(selection_map, tmp_df['y_ind'], tmp_df['x_ind'], intensity)

        else:

            self.set_map_values_to_green(selection_map, df['y_ind'], df['x_ind'])

        return selection_map

    @staticmethod
    def orientation_map(experiment_id, orientation_rule, ax):

        if orientation_rule == 'parallel_to_short_wall':
            if experiment_id in ('exp_scales_a', 'exp_scales_c'):
                x = 0.5
                y = 0
            else:
                x = 0
                y = 0.5
        elif orientation_rule == 'orthogonal_to_short_wall':
            if experiment_id in ('exp_scales_a', 'exp_scales_c'):
                x = 0
                y = 0.5
            else:
                x = 0.5
                y = 0
        else:
            raise ValueError('Unknown orientation_rule'.format(orientation_rule))

        ax.arrow(0, 0, x, y, head_width=0.5, head_length=0.5, width=0.1, linewidth=0, color=(0, 0.5, 0))
        ax.arrow(0, 0, -x, -y, head_width=0.5, head_length=0.5, width=0.1, linewidth=0, color=(0, 0.5, 0))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal', 'box')
        ax.axis('off')

    @staticmethod
    def plot_direction(experiment_id, direction_rule, ax):

        if direction_rule == 'parallel_to_short_wall':
            if experiment_id in ('exp_scales_a', 'exp_scales_c'):
                rotation = 90
            else:
                rotation = 0
        elif direction_rule == 'orthogonal_to_short_wall':
            if experiment_id in ('exp_scales_a', 'exp_scales_c'):
                rotation = 0
            else:
                rotation = 90
        else:
            raise ValueError('Unknown direction_rule'.format(direction_rule))

        ax.add_patch(matplotlib.patches.Wedge(0, 1, 45 + rotation, 135 + rotation, color=(0, 0.5, 0)))
        ax.add_patch(matplotlib.patches.Wedge(0, 1, 225 + rotation, 315 + rotation, color=(0, 0.5, 0)))
        ax.add_patch(matplotlib.patches.Wedge(0, 1, 315 + rotation, 45 + rotation, color=(0.7, 0.7, 0.7)))
        ax.add_patch(matplotlib.patches.Wedge(0, 1, 135 + rotation, 225 + rotation, color=(0.7, 0.7, 0.7)))
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal', 'box')
        ax.axis('off')

    def draw_legend(self, ax_map, ax_angle, experiment_id, distance_measure=None, distance_bin_width=None,
                    filter_to_middle_third=False, filter_to_middle_fifth=False, filter_to_first_bin_from_wall=False,
                    orientation_rule=None, direction_rule=None, max_bin_center=None):

        if not (orientation_rule is None):
            self.orientation_map(experiment_id, orientation_rule, ax_angle)

        if not (direction_rule is None):
            self.plot_direction(experiment_id, direction_rule, ax_angle)

        ax_map.imshow(
            self.location_map(
                experiment_id, distance_measure=distance_measure, distance_bin_width=distance_bin_width,
                filter_to_middle_third=filter_to_middle_third, filter_to_middle_fifth=filter_to_middle_fifth,
                filter_to_first_bin_from_wall=filter_to_first_bin_from_wall, max_bin_center=max_bin_center
            )
        )
        ax_map.axis('off')

    def append_to_axes(self, ax, experiment_id, distance_measure=None, distance_bin_width=None,
                       filter_to_middle_third=False, filter_to_middle_fifth=False,
                       filter_to_first_bin_from_wall=False,
                       orientation_rule=None, direction_rule=None,
                       proportional_to_environment_size=False, max_bin_center=None):

        if proportional_to_environment_size:
            if orientation_rule is None and direction_rule is None:
                map_and_angle_height = {'exp_scales_a': '10%',
                                        'exp_scales_b': '10%',
                                        'exp_scales_c': '20%',
                                        'exp_scales_d': '20%'}[experiment_id]
            else:
                map_and_angle_height = {'exp_scales_a': '7%',
                                        'exp_scales_b': '7%',
                                        'exp_scales_c': '14%',
                                        'exp_scales_d': '14%'}[experiment_id]
        else:
            map_and_angle_height = '10%'

        ax_title = None
        if not (orientation_rule is None) or not (direction_rule is None):

            if map_and_angle_height is None:
                ax_title = inset_axes(ax, width="100%", height="1%",
                                      bbox_to_anchor=(0, 1.025, 1/3., 0.001),
                                      bbox_transform=ax.transAxes, loc='lower center', borderpad=0)

            ax_map = inset_axes(ax, width="100%", height=map_and_angle_height,
                                bbox_to_anchor=((1/3., 1.025, 1/3., 1)
                                                if map_and_angle_height is None
                                                else (0, 1.025, 0.5, 1)),
                                bbox_transform=ax.transAxes, loc='lower center', borderpad=0)

            ax_angle = inset_axes(ax, width="100%", height=map_and_angle_height,
                                  bbox_to_anchor=((2/3., 1.025, 1/3., 1)
                                                  if map_and_angle_height is None
                                                  else (0.5, 1.025, 0.5, 1)),
                                  bbox_transform=ax.transAxes, loc='lower center', borderpad=0)

        else:

            if map_and_angle_height is None:
                ax_title = inset_axes(ax, width="100%", height="1%",
                                      bbox_to_anchor=(0, 1.025, 0.5, 0.001),
                                      bbox_transform=ax.transAxes, loc='lower center', borderpad=0)

            ax_map = inset_axes(ax, width="100%", height=map_and_angle_height,
                                bbox_to_anchor=((0.5, 1.025, 0.5, 1)
                                                if map_and_angle_height is None
                                                else (0, 1.025, 1, 1)),
                                bbox_transform=ax.transAxes, loc='lower center', borderpad=0)
            ax_angle = None

        if ax_title is not None and map_and_angle_height is None:
            ax_title.set_title('({})'.format(experiment_id_substitutes[experiment_id]))
            ax_title.axis('off')
        else:
            ax_map.text(0.5, 0.5, r'$\bf{}$'.format(experiment_id_substitutes[experiment_id]),
                        va='center', ha='center', transform=ax_map.transAxes)

        self.draw_legend(
            ax_map, ax_angle, experiment_id, distance_measure=distance_measure,
            distance_bin_width=distance_bin_width, filter_to_middle_third=filter_to_middle_third,
            filter_to_middle_fifth=filter_to_middle_fifth,
            filter_to_first_bin_from_wall=filter_to_first_bin_from_wall,
            orientation_rule=orientation_rule, direction_rule=direction_rule,
            max_bin_center=max_bin_center
        )


spatial_filter_legend_instance = SpatialFilteringLegend()


def add_stat_annotation_with_n_vals(ax, df, x, y, order=None, hue=None, box_pairs=None,
                                    test='Mann-Whitney', loc='inside',
                                    pvalue_thresholds=pvalue_thresholds, text_va='center',
                                    cut_kde_to_95_percentile=False, apply_bonferroni_correction=False, **kwargs):

    _, results = add_stat_annotation(
        ax, data=df, x=x, y=y, order=order, hue=hue, box_pairs=box_pairs, test=test,
        text_format='star', loc=loc, pvalue_thresholds=pvalue_thresholds,
        line_offset=0.0025, line_height=0.01, text_offset=0, text_va=text_va, verbose=0,
        cut_kde_to_95_percentile=cut_kde_to_95_percentile, apply_bonferroni_correction=apply_bonferroni_correction,
        **kwargs
    )

    for result in results:

        if hue is None:
            idx_1 = df[x] == result['box1']
            idx_2 = df[x] == result['box2']
        else:
            idx_1 = (df[x] == result['box1'][0]) & (df[hue] == result['box1'][1])
            idx_2 = (df[x] == result['box2'][0]) & (df[hue] == result['box2'][1])

        result['N'] = ('box1', np.sum(idx_1), 'box2', np.sum(idx_2))

    return results


def get_max_ylim(axs):

    ylim = (np.inf, -np.inf)
    for ax in axs:
        ylim = [ax.get_ylim()[0] if ax.get_ylim()[0] < ylim[0] else ylim[0],
                ax.get_ylim()[1] if ax.get_ylim()[1] > ylim[1] else ylim[1]]

    return ylim


def plot_stats_dict_to_axes(test_results, stat_ax, loc=(0, 1), va='top',
                            wrapchar=75, fontsize='small', crop_results=True):

    if isinstance(test_results, str):
        stat_ax.text(*loc, textwrap.wrap(test_results, wrapchar), ha='left', va=va, color='black', fontsize=fontsize)
        return

    if crop_results:
        crop_statannot_results(test_results)

    stats_text = '\n'.join([pprint.pformat(x, indent=4, width=wrapchar, compact=True) for x in test_results])
    stat_ax.text(*loc, stats_text, ha='left', va=va, color='black', fontsize=fontsize)


def add_stat_annotation_with_n_vals(ax, df, x, y, order=None, hue=None, box_pairs=None,
                                    test='Mann-Whitney', loc='inside',
                                    pvalue_thresholds=pvalue_thresholds, text_va='center',
                                    cut_kde_to_95_percentile=False, apply_bonferroni_correction=False, **kwargs):

    _, results = add_stat_annotation(
        ax, data=df, x=x, y=y, order=order, hue=hue, box_pairs=box_pairs, test=test,
        text_format='star', loc=loc, pvalue_thresholds=pvalue_thresholds,
        line_offset=0.0025, line_height=0.01, text_offset=0, text_va=text_va, verbose=0,
        cut_kde_to_95_percentile=cut_kde_to_95_percentile, apply_bonferroni_correction=apply_bonferroni_correction,
        **kwargs
    )

    for result in results:

        if hue is None:
            idx_1 = df[x] == result['box1']
            idx_2 = df[x] == result['box2']
        else:
            idx_1 = (df[x] == result['box1'][0]) & (df[hue] == result['box1'][1])
            idx_2 = (df[x] == result['box2'][0]) & (df[hue] == result['box2'][1])

        result['N'] = ('box1', np.sum(idx_1), 'box2', np.sum(idx_2))

    return results


def plot_raincloud(ax, x, y, df, groups_order=None, ort='v', whis=(5, 95), hue=None, hue_order=None,
                   palette=None, data_selection_label_kwargs=None, bound_kda_by_95_percentiles=True,
                   stripplot_size=1):

    ptitprince.half_violinplot(x=x, y=y, data=df, order=groups_order, hue=hue, hue_order=hue_order,
                               bw=.2, cut=0., scale="area", width=.6, palette=palette,
                               inner=None, orient=ort, bound_kda_by_95_percentiles=bound_kda_by_95_percentiles,
                               ax=ax)
    sns.boxplot(x=x, y=y, data=df, order=groups_order,  hue=hue, hue_order=hue_order, color='black',
                width=.15, zorder=10, showcaps=True, whis=whis,
                boxprops={'facecolor': 'none', 'zorder': 10}, showfliers=False,
                whiskerprops={'linewidth': 2, 'zorder': 10},
                saturation=1, orient=ort, ax=ax)
    ylim = ax.get_ylim()
    sns.stripplot(x=x, y=y, data=df, order=groups_order, hue=hue, hue_order=hue_order,
                  orient=ort, palette=palette, ax=ax, size=stripplot_size, zorder=-1)
    ax.set_ylim(ylim)
    ax.set_xlabel('')

    if not (data_selection_label_kwargs is None):
        spatial_filter_legend_instance.append_to_axes(
            ax, **data_selection_label_kwargs
        )


def plot_raincloud_and_stats(x, y, df, ax, stat_ax, x_order, pairwise_stats_on_plot=True,
                             pairwise_test='Mann-Whitney', data_selection_label_kwargs=None,
                             cut_kde_to_95_percentile=False, group_test_bonferroni_m=1,
                             apply_bonferroni_correction_to_pairwise_tests=False, palette=None,
                             stripplot_size=1):

    plot_raincloud(ax, x, y, df, groups_order=x_order, ort='v', whis=(5, 95), palette=palette,
                   data_selection_label_kwargs=data_selection_label_kwargs,
                   stripplot_size=stripplot_size)

    kruskal_h_value, kruskal_pvalue = \
        kruskal(*[df[df[x] == group][y] for group in x_order])
    kruskal_pvalue = min(kruskal_pvalue * group_test_bonferroni_m, 1.0)
    levene_w_value, levene_pvalue = \
        levene(*[df[df[x] == group][y] for group in x_order], center='median')
    levene_pvalue = min(levene_pvalue * group_test_bonferroni_m, 1.0)

    if pairwise_test == 'Mann-Whitney':
        multi_group_pvalue = kruskal_pvalue
    elif pairwise_test == 'Levene':
        multi_group_pvalue = levene_pvalue
    else:
        raise ValueError('Unknown pairwise_test argument {}'.format(pairwise_test))

    if multi_group_pvalue < 0.05:
        if pairwise_stats_on_plot:
            test_results = add_stat_annotation_with_n_vals(
                ax, df, x, y, order=x_order, box_pairs=list(combinations(x_order, 2)),
                test=pairwise_test, cut_kde_to_95_percentile=cut_kde_to_95_percentile,
                stats_params=({'center': 'median'} if pairwise_test == 'Levene' else {}),
                apply_bonferroni_correction=apply_bonferroni_correction_to_pairwise_tests
            )
        else:
            test_results = compute_pairwise_comparisons(
                df, x, y, list(combinations(x_order, 2)), test=pairwise_test,
                apply_bonferroni_correction=apply_bonferroni_correction_to_pairwise_tests
            )
    else:
        test_results = 'Multi group test p value >= 0.05'

    if pairwise_test == 'Levene':

        y_points = []
        for x_bin in x_order:
            s = df.loc[df[x] == x_bin, y]
            m = s.median()
            v = s.std()
            y_points.append((m - v, m + v))

        y_points = np.array(y_points)

        patch_xy = np.stack((np.concatenate((np.arange(len(x_order)), np.arange(len(x_order))[::-1])),
                             np.concatenate((y_points[:, 0], y_points[::-1, 1]))),
                            axis=1)

        ax.add_patch(matplotlib.patches.Polygon(patch_xy, alpha=0.25, color='gray'))

    table_cell_text = [['', 'statistic', 'p-value'],
                       ['Kruskal-Wallis test', '{:.2e}'.format(kruskal_h_value), '{:.2e}'.format(kruskal_pvalue)],
                       ['', 'statistic', 'p-value'],
                       ['Levene test', '{:.2e}'.format(levene_w_value), '{:.2e}'.format(levene_pvalue)], ['', '', ''],
                       ['group', 'nanmean', '']]

    for group in x_order:
        table_cell_text.append([str(group), str(np.nanmean(df.loc[df[x] == group, y])), ''])

    stat_ax.table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')
    plot_stats_dict_to_axes(test_results, stat_ax, loc=(0, 0), va='bottom')


def maps_by_environment_to_df_with_distances(maps, bin_size=Params.spatial_ratemap['bin_size'], drop_nans=True,
                                             stack_group_name=None, stack_groups=None):

    bin_distance_to_wall = snippets.SpatialMapBinCenterDistanceToWall()

    dfs = []
    xy_of_bin_centres_list = []
    for experiment_id, value_map in maps.items():

        distance_to_wall = []
        values = []
        n_xbins = []
        n_ybins = []
        for n_xbin in range(value_map.shape[1]):
            for n_ybin in range(value_map.shape[0]):
                d = bin_distance_to_wall.get(
                    value_map.shape[:2], spatial_windows[experiment_id], n_xbin, n_ybin
                )
                distance_to_wall.append(d * np.ones(value_map.shape[2]))
                if isinstance(value_map, np.ma.core.MaskedArray):
                    values.append(value_map.data[n_ybin, n_xbin, ...])
                else:
                    values.append(value_map[n_ybin, n_xbin, ...])
                n_xbins += [n_xbin] * value_map.shape[2]
                n_ybins += [n_ybin] * value_map.shape[2]

        distance_to_wall = np.concatenate(distance_to_wall)
        values = np.concatenate(values)

        xy_of_bin_centres = \
            (np.stack((n_xbins, n_ybins), axis=1).astype(np.float32) + 0.5) * bin_size
        xy_of_bin_centres_list.append(xy_of_bin_centres)

        dfs.append(pd.DataFrame(
            {
                'distance to wall (cm)': distance_to_wall,
                'value': values,
                'experiment_id': [experiment_id] * values.size
            }
        ))

        if not (stack_group_name is None) and not (stack_groups is None):
            stack_group_values = stack_groups if isinstance(stack_groups, list) else stack_groups[experiment_id]
            dfs[-1][stack_group_name] = \
                list(np.concatenate([stack_group_values] * (value_map.shape[0] * value_map.shape[1])))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    xy_of_bin_centres = np.concatenate(xy_of_bin_centres_list, axis=0)

    compute_distances_to_landmarks(df, xy_of_bin_centres)

    df.replace(to_replace={'experiment_id': experiment_id_substitutes}, inplace=True)
    df.rename(columns={'experiment_id': 'environment'}, inplace=True)

    if drop_nans:
        df.dropna(inplace=True)

    return df


def get_field_data_with_distance_to_boundary(all_recordings, df_units, df_fields,
                                             property_names, experiment_ids, verbose=False):

    if isinstance(property_names, str):
        property_names = (property_names,)

    all_property_maps_list = []
    for property_name in property_names:
        all_property_maps_list.append(PlaceFieldPropertyMaps(
            all_recordings, df_units, df_fields, unit_category='place_cell',
            property_name=property_name, verbose=verbose
        ))

    dfs = []
    for property_name, all_property_maps in zip(property_names, all_property_maps_list):

        maps = {}
        maps_animal_ids = {}
        for experiment_id in experiment_ids:

            # Collect arrays per animal
            property_maps = []
            animal_ids = []
            for animal in Params.animal_ids:

                property_maps.append(all_property_maps.get_maps_for_animal(property_name, experiment_id, animal))
                animal_ids.append([animal] * property_maps[-1].shape[2])

            maps[experiment_id] = np.ma.concatenate(property_maps, axis=2)
            maps_animal_ids[experiment_id] = sum(animal_ids, [])

        df = maps_by_environment_to_df_with_distances(
            maps, bin_size=Params.spatial_ratemap['bin_size'], drop_nans=False,
            stack_group_name='animal', stack_groups=maps_animal_ids
        )
        dfs.append(df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        for df, property_name in zip(dfs, property_names):
            df.rename(columns={'value': property_name}, inplace=True)
        for df, property_name in zip(dfs[1:], property_names[1:]):
            dfs[0][property_name] = df[property_name]
        df = dfs[0]

    return df


def compute_pairwise_comparisons(df, grouping_variable, test_variable, group_pairs,
                                 second_grouping_variable=None, test='Mann-Whitney',
                                 apply_bonferroni_correction=False):

    if test == 'Mann-Whitney':

        test_func = mannwhitneyu

    elif test == 'Wilcoxon':

        test_func = wilcoxon

    elif test == 'Levene':

        test_func = levene

    else:

        raise ValueError('Unknown test {}'.format(test))

    results = []
    for g1, g2 in group_pairs:

        if second_grouping_variable is None:
            stat, pvalue = test_func(df.loc[df[grouping_variable] == g1, test_variable],
                                     df.loc[df[grouping_variable] == g2, test_variable])
        else:
            stat, pvalue = test_func(df.loc[(df[grouping_variable] == g1[0])
                                            & (df[second_grouping_variable] == g1[1]), test_variable],
                                     df.loc[(df[grouping_variable] == g2[0])
                                            & (df[second_grouping_variable] == g2[1]), test_variable])

        if apply_bonferroni_correction:
            pvalue = min(pvalue * len(group_pairs), 1.0)

        if second_grouping_variable is None:
            n1 = np.sum(df[grouping_variable] == g1)
            n2 = np.sum(df[grouping_variable] == g2)
        else:
            n1 = np.sum((df[grouping_variable] == g1[0]) & (df[second_grouping_variable] == g1[1]))
            n2 = np.sum((df[grouping_variable] == g2[0]) & (df[second_grouping_variable] == g2[1]))

        results.append(
            {'info': {'test': test, 'groups': (g1, g2), 'n': (n1, n2)},
             'statistic': stat,
             'p-value': pvalue}
        )

    return results


def crop_statannot_results(test_results):
    for test_result in test_results:
        if 'pvalue' in test_result:
            del test_result['pvalue']
        if 'test_short_name' in test_result:
            del test_result['test_short_name']


class ValueByBinnedMeasurePlot(object):

    def __init__(self, df, value, binned_measure, fig, ax, stat_axs, kind='box', hue=None, hue_order=None,
                 first_plot_title=None, xlabel=None, ylabel=None,
                 friedman_grouping_variable=None, pairwise_stats_on_plot=True,
                 line_offset_to_box=None, ylim='auto', yscale=None, connect_the_dots_by_animal=True,
                 plot_environment_comparison_by_first_bin=True, split=False, legend_environment=None,
                 environments_in_different_colors=True, plot_stats_test=None):

        self._df = df
        self._value = value
        self._binned_measure = binned_measure
        self._fig = ax.figure
        self._ax = ax
        self._stat_axs = stat_axs
        self._kind = kind
        self._hue = hue
        self._hue_order = hue_order
        self._first_plot_title = first_plot_title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._friedman_grouping_variable = friedman_grouping_variable
        self._pairwise_stats_on_plot = pairwise_stats_on_plot
        self._line_offset_to_box = line_offset_to_box
        self._input_ylim = ylim
        self._yscale = yscale
        self._connect_the_dots_by_animal = connect_the_dots_by_animal
        self._plot_environment_comparison_by_first_bin = plot_environment_comparison_by_first_bin
        self._split = split
        self._legend_environment = legend_environment
        self._plot_stats_test = plot_stats_test
        if self._plot_stats_test is None:
            self._plot_stats_test = 'Mann-Whitney' if self._friedman_grouping_variable is None else 'Wilcoxon'

        self._binned_measure_order = sorted(self._df[self._binned_measure].unique())

        self._environments = sorted(self._df['environment'].unique())

        self._binned_measure_order = {}
        for environment in self._environments:
            self._binned_measure_order[environment] = \
                sorted(self._df.loc[self._df['environment'] == environment, self._binned_measure].unique())
        self._first_bin = self._binned_measure_order[environment][0]

        if environments_in_different_colors:
            self._colors = sns.color_palette(n_colors=len(self._environments))
            self._colors_dict = {key: color for key, color in zip(self._environments, self._colors)}
        else:
            self._colors_dict = {key: sns.color_palette(n_colors=1)[0] for key in self._environments}

        self._axs = None
        self._ylim = None

        # Prepare add_stat_annotation parameters

        if not (self._line_offset_to_box is None):
            self._stat_annotation_loc = 'outside'
            self._use_fixed_offset = True
        else:
            self._stat_annotation_loc = 'inside'
            self._use_fixed_offset = False

        # Create stats output variables

        self._kruskal_h_value_first_bin = np.nan
        self._kruskal_pvalue_first_bin = np.nan
        self._friedman_chisq_value_first_bin = np.nan
        self._friedman_pvalue_first_bin = np.nan
        self._mw_tests_first_bin = []

        self._kruskal_h_values = [np.nan] if plot_environment_comparison_by_first_bin else []
        self._kruskal_pvalues = [np.nan] if plot_environment_comparison_by_first_bin else []
        self._friedman_chisq_values = [np.nan] if plot_environment_comparison_by_first_bin else []
        self._friedman_pvalues = [np.nan] if plot_environment_comparison_by_first_bin else []
        self._mw_tests = []

        # Plot all axes
        self.plot()

    @property
    def axs(self):
        return self._axs

    def create_axes(self):

        if self._plot_environment_comparison_by_first_bin:

            width_ratios = [4, 0.5] + [len(self._binned_measure_order[env]) for env in self._environments[1:]]
            gs = GridSpecFromSubplotSpec(1, len(width_ratios), self._ax, wspace=0.15, width_ratios=width_ratios)
            self._ax.axis('off')
            self._axs = [self._fig.add_subplot(g) for g in gs]
            ax_empty = self._axs.pop(1)
            ax_empty.axis('off')

        else:

            width_ratios = [len(self._binned_measure_order[env]) for env in self._environments]
            gs = GridSpecFromSubplotSpec(1, len(width_ratios), self._ax, wspace=0.15, width_ratios=width_ratios)
            self._ax.axis('off')
            self._axs = [self._fig.add_subplot(g) for g in gs]

    def plot_comparison_of_first_bin_between_environments(self):

        if self._kind in ('box', 'strip'):

            if self._kind == 'box':

                sns.boxplot(x='environment', y=self._value, order=self._environments,
                            hue=self._hue, hue_order=self._hue_order,
                            data=self._df[self._df[self._binned_measure] == self._first_bin],
                            ax=self._axs[0], whis='range')

            sns.stripplot(x='environment', y=self._value, order=self._environments,
                          hue=self._hue, hue_order=self._hue_order,
                          data=self._df[self._df[self._binned_measure] == self._first_bin],
                          ax=self._axs[0], linewidth=1, zorder=3, dodge=True)

            if self._connect_the_dots_by_animal and self._hue is None:
                for animal in self._df['animal'].unique():
                    y_vals = []
                    for environment in self._environments:
                        y_vals.append(
                            self._df.loc[(self._df['animal'] == animal)
                                         & (self._df['environment'] == environment)
                                         & (self._df[self._binned_measure] == self._first_bin),
                                         self._value].values[0]
                        )
                    self._axs[0].plot(np.arange(len(self._environments)), y_vals, linewidth=2,
                                      color='gray', alpha=0.75, zorder=2)

        elif self._kind == 'violin':

            sns.violinplot(x='environment', y=self._value, order=self._environments,
                           hue=self._hue, hue_order=self._hue_order,
                           data=self._df[self._df[self._binned_measure] == self._first_bin],
                           cut=0., ax=self._axs[0], split=self._split, legend=(None if self._hue is None else False),
                           scale='width')

        else:

            raise ValueError('Unknown kind {}'.format(self._kind))

        if self._first_plot_title is None:
            self._axs[0].set_title('bin {}'.format(self._first_bin))
        else:
            self._axs[0].set_title(self._first_plot_title)

        self._axs[0].set_xlabel('environment')

    def plot_comparison_of_measure_bins_within_environments(self):

        axs = self._axs[1:] if self._plot_environment_comparison_by_first_bin else self._axs
        environments = self._environments[1:] if self._plot_environment_comparison_by_first_bin else self._environments
        palette = (sns.color_palette(n_colors=len(self._environments))[1:]
                   if self._plot_environment_comparison_by_first_bin
                   else sns.color_palette(n_colors=len(self._environments)))

        for ax, environment, color in zip(axs, environments, palette):

            if self._kind in ('box', 'strip'):

                if self._kind == 'box':

                    sns.boxplot(x=self._binned_measure, y=self._value, order=self._binned_measure_order[environment],
                                hue=self._hue, hue_order=self._hue_order,
                                data=self._df[self._df['environment'] == environment],
                                ax=ax, whis='range', color=self._colors_dict[environment])

                sns.stripplot(x=self._binned_measure, y=self._value, order=self._binned_measure_order[environment],
                              hue=self._hue, hue_order=self._hue_order,
                              data=self._df[self._df['environment'] == environment],
                              ax=ax, color=color, linewidth=1, zorder=3, dodge=True)

                if self._connect_the_dots_by_animal and self._hue is None:
                    for animal in self._df['animal'].unique():
                        y_vals = []
                        for binned_measure_value in self._binned_measure_order[environment]:
                            y_vals.append(
                                self._df.loc[(self._df['animal'] == animal) & (self._df['environment'] == environment)
                                             & (self._df[self._binned_measure] == binned_measure_value),
                                             self._value].values[0]
                            )
                        ax.plot(np.arange(len(self._binned_measure_order[environment])), y_vals, linewidth=2,
                                color='gray', alpha=0.75, zorder=2)

            elif self._kind == 'violin':

                sns.violinplot(x=self._binned_measure, y=self._value, order=self._binned_measure_order[environment],
                               hue=self._hue, hue_order=self._hue_order,
                               data=self._df[self._df['environment'] == environment],
                               cut=0., ax=ax, color=self._colors_dict[environment],
                               split=self._split, scale='width')

            else:

                raise ValueError('Unknown kind {}'.format(self._kind))

            if not (self._hue_order is None) and environment == self._legend_environment:

                handles, labels = ax.get_legend_handles_labels()
                if self._kind == 'box':
                    ax.legend(handles[len(self._hue_order):], labels[len(self._hue_order):])
                elif self._kind == 'violin':
                    ax.legend(handles, labels)

            elif not (self._hue_order is None):

                ax.legend().set_visible(False)

            # Format plots
            ax.set_title('( {} )'.format(environment))
            ax.set_ylabel('')
            ax.set_xlabel('')

            if self._plot_environment_comparison_by_first_bin or not (ax is axs[0]):
                ax.set_yticklabels([])

    def format_across_axes(self):

        if not (self._ylabel is None):
            self._axs[0].set_ylabel(self._ylabel)

        if not (self._xlabel is None):
            self._axs[2].set_xlabel(self._xlabel)

        if not (self._yscale is None):
            for ax in self._axs:
                ax.set_yscale(self._yscale)
            for ax in self._axs[1:]:
                ax.set_yticklabels([])

        # Set ylim equal between all plots
        if self._input_ylim == 'auto':
            self._ylim = get_max_ylim(self._axs)
        else:
            self._ylim = self._input_ylim

        if not (self._ylim is None):
            for ax in self._axs:
                ax.set_ylim(self._ylim)

    def compute_statistics_for_first_bin_across_environments(self):

        if not (self._friedman_grouping_variable is None):
            self._df = self._df.sort_values(self._friedman_grouping_variable)

        hue_criteria = True if self._hue is None else self._df[self._hue] == self._hue_order[1]
        test_list = [self._df[(self._df['environment'] == environment)
                              & (self._df[self._binned_measure] == self._first_bin) & hue_criteria][self._value]
                     for environment in self._environments]
        self._kruskal_h_value_first_bin, self._kruskal_pvalue_first_bin = kruskal(*test_list)

        if not (self._friedman_grouping_variable is None):
            self._friedman_chisq_value_first_bin, self._friedman_pvalue_first_bin = friedmanchisquare(*test_list)

        if self._hue is None:
            box_pairs = list(combinations(self._environments, 2))
        else:
            box_pairs = [((environment, self._hue_order[0]), (environment, self._hue_order[1]))
                         for environment in self._environments]

        if not (self._friedman_grouping_variable is None):
            group_test_significant = self._friedman_pvalue_first_bin < 0.05
        else:
            group_test_significant = self._kruskal_pvalue_first_bin < 0.05

        if self._pairwise_stats_on_plot and group_test_significant:

            tmp_stat_results = add_stat_annotation_with_n_vals(
                self._axs[0], self._df[self._df[self._binned_measure] == self._first_bin], 'environment',
                self._value, order=self._environments, hue=self._hue, hue_order=self._hue_order, box_pairs=box_pairs,
                use_fixed_offset=self._use_fixed_offset, line_offset_to_box=self._line_offset_to_box,
                text_va=('center' if self._hue is None else 'bottom'),
                pvalue_thresholds=(pvalue_thresholds if self._hue is None
                                   else [[0.001, '***'], [0.01, '**'], [0.05, '*'], [1, 'ns']]),
                test=self._plot_stats_test
            )
            crop_statannot_results(tmp_stat_results)
            self._mw_tests_first_bin.append(tmp_stat_results)

        self._mw_tests_first_bin.append(compute_pairwise_comparisons(
            self._df[self._df[self._binned_measure] == self._first_bin], 'environment', self._value,
            box_pairs, second_grouping_variable=self._hue, test='Mann-Whitney'
        ))

        if not (self._friedman_grouping_variable is None):

            tmp = compute_pairwise_comparisons(
                self._df[self._df[self._binned_measure] == self._first_bin], 'environment', self._value,
                box_pairs, second_grouping_variable=self._hue, test='Wilcoxon'
            )
            self._mw_tests_first_bin.append(tmp)

    def compute_statistics_across_measure_bins_in_each_environment(self):

        axs = self._axs[1:] if self._plot_environment_comparison_by_first_bin else self._axs
        environments = self._environments[1:] if self._plot_environment_comparison_by_first_bin else self._environments

        for ax, environment in zip(axs, environments):

            if len(self._binned_measure_order[environment]) > 2:

                hue_criteria = True if self._hue is None else self._df[self._hue] == self._hue_order[1]

                test_list = [self._df[(self._df['environment'] == environment)
                                      & (self._df[self._binned_measure] == measure_bin) & hue_criteria][self._value]
                             for measure_bin in self._binned_measure_order[environment]]
                a, b = kruskal(*test_list)
                self._kruskal_h_values.append(a)
                self._kruskal_pvalues.append(b)

                if not (self._friedman_grouping_variable is None):
                    c, d = friedmanchisquare(*test_list)
                    self._friedman_chisq_values.append(c)
                    self._friedman_pvalues.append(d)

            else:

                self._kruskal_h_values.append(np.nan)
                self._kruskal_pvalues.append(np.nan)
                self._friedman_chisq_values.append(np.nan)
                self._friedman_pvalues.append(np.nan)

            if self._hue is None:
                box_pairs = list(combinations(self._binned_measure_order[environment], 2))
            else:
                box_pairs = [((binned_measure, self._hue_order[0]), (binned_measure, self._hue_order[1]))
                             for binned_measure in self._binned_measure_order[environment]]

            if self._pairwise_stats_on_plot:

                if not (self._friedman_grouping_variable is None):
                    group_test_significant = self._friedman_pvalues[-1] < 0.05
                else:
                    group_test_significant = self._kruskal_pvalues[-1] < 0.05

                if (len(self._binned_measure_order[environment]) == 2 or group_test_significant
                        or not (self._hue is None)):

                    tmp = add_stat_annotation_with_n_vals(
                        ax, self._df[self._df['environment'] == environment], self._binned_measure, self._value,
                        order=self._binned_measure_order[environment], hue=self._hue, hue_order=self._hue_order,
                        box_pairs=box_pairs, loc=self._stat_annotation_loc,
                        use_fixed_offset=self._use_fixed_offset, line_offset_to_box=self._line_offset_to_box,
                        text_va=('center' if self._hue is None else 'bottom'),
                        pvalue_thresholds=(pvalue_thresholds if self._hue is None
                                           else [[0.001, '***'], [0.01, '**'], [0.05, '*'], [1, 'ns']]),
                        test=self._plot_stats_test
                    )
                    crop_statannot_results(tmp)
                    self._mw_tests.append(tmp)

            self._mw_tests.append(compute_pairwise_comparisons(
                self._df[self._df['environment'] == environment], self._binned_measure, self._value,
                box_pairs, second_grouping_variable=self._hue, test='Mann-Whitney'
            ))

            if not (self._friedman_grouping_variable is None):
                self._mw_tests.append(compute_pairwise_comparisons(
                    self._df[self._df['environment'] == environment], self._binned_measure, self._value,
                    box_pairs, second_grouping_variable=self._hue, test='Wilcoxon'
                ))

    def output_stats_on_stat_axs(self):

        if self._stat_axs is None:
            return

        self._stat_axs[0].set_title('measure {} | value {}'.format(self._binned_measure, self._value))
        self._stat_axs[1].set_title('measure {} | value {}'.format(self._binned_measure, self._value))
        table_cell_text = [['First bin', '', ''],
                           ['', 'H-value', 'p-value'],
                           ['Kruskal-Wallis test',
                            '{:.2e}'.format(self._kruskal_h_value_first_bin),
                            '{:.2e}'.format(self._kruskal_pvalue_first_bin)],
                           ['', '', '']]
        if not (self._friedman_grouping_variable is None):
            table_cell_text += [
                ['Friedman test',
                 '{:.2e}'.format(self._friedman_chisq_value_first_bin),
                 '{:.2e}'.format(self._friedman_pvalue_first_bin)],
                ['', '', '']
            ]

        if not (self._friedman_grouping_variable is None):
            for a, b, c, d, environment in zip(self._kruskal_h_values, self._kruskal_pvalues,
                                               self._friedman_chisq_values, self._friedman_pvalues,
                                               self._environments):
                table_cell_text += [
                    ['Environment', environment, ''],
                    ['', 'H-value', 'p-value'],
                    ['Kruskal-Wallis test', '{:.2e}'.format(a), '{:.2e}'.format(b)],
                    ['Friedman test', '{:.2e}'.format(c), '{:.2e}'.format(d)],
                    ['', '', '']
                ]
        else:
            for a, b, environment in zip(self._kruskal_h_values, self._kruskal_pvalues, self._environments):
                table_cell_text += [
                    ['Environment', environment, ''],
                    ['', 'H-value', 'p-value'],
                    ['Kruskal-Wallis test', '{:.2e}'.format(a), '{:.2e}'.format(b)],
                    ['', '', '']
                ]

        self._stat_axs[0].table(cellText=table_cell_text, cellLoc='left', loc='upper left', edges='open')
        plot_stats_dict_to_axes((self._mw_tests_first_bin, self._mw_tests), self._stat_axs[1],
                                wrapchar=200, fontsize='x-small', crop_results=False)

    def plot(self):

        self.create_axes()

        if self._plot_environment_comparison_by_first_bin:
            self.plot_comparison_of_first_bin_between_environments()

        self.plot_comparison_of_measure_bins_within_environments()

        self.format_across_axes()  # This must be done before stats in case stats are needed outside

        if self._plot_environment_comparison_by_first_bin:
            self.compute_statistics_for_first_bin_across_environments()

        self.compute_statistics_across_measure_bins_in_each_environment()

        self.format_across_axes()  # redo, because ylim values may have shifted after using `add_stat_annotation`

        self.output_stats_on_stat_axs()

    @property
    def ylim(self):
        return deepcopy(self._ylim)

    def set_ylim(self, ylim):
        self._ylim = ylim
        for ax in self._axs:
            ax.set_ylim(self._ylim)


class ValueByBinnedDistancePlot(ValueByBinnedMeasurePlot):

    distance_bin_width = 28

    long_wall_proportional_distance_middle_third_filter = 2. / 3.

    long_wall_proportional_distance_middle_fifth_filter = 4. / 5.

    def __init__(self, df, value, binned_measure, *args, hue=None, hue_order=None,
                 filter_to_middle_third=False, filter_to_first_bin_from_wall=False,
                 aggregate_by_distance_and_animal=None, data_selection_label_kwargs=None,
                 weight_values_per_bin_by_mean_weight=None, **kwargs):

        columns = [value, binned_measure, 'environment'] + ([] if hue is None else [hue])
        if binned_measure != 'distance to wall (cm)':
            columns.append('distance to wall (cm)')
        if filter_to_middle_third:
            columns.append('distance_to_long_wall_prop')
        if not (aggregate_by_distance_and_animal is None):
            columns.append('animal')
        if not (weight_values_per_bin_by_mean_weight is None):
            columns.append(weight_values_per_bin_by_mean_weight)

        df = df[columns].copy(deep=True)
        df.dropna(inplace=True)

        ValueByBinnedDistancePlot.bin_distance_values(df, binned_measure,
                                                      ValueByBinnedDistancePlot.distance_bin_width)

        if filter_to_middle_third:
            df = ValueByBinnedDistancePlot.filter_to_middle_third(df)

        if filter_to_first_bin_from_wall:
            df = ValueByBinnedDistancePlot.filter_to_first_bin_from_wall(df)

        if not (weight_values_per_bin_by_mean_weight is None):
            agg_columns = ['environment', binned_measure] + ([] if hue is None else [hue])
            df_weights = df.groupby(agg_columns)[weight_values_per_bin_by_mean_weight].mean().reset_index()
            del df[weight_values_per_bin_by_mean_weight]
            df = df.merge(df_weights, how='left', on=agg_columns)
            df[value] = df[value] / df[weight_values_per_bin_by_mean_weight]

        if not (aggregate_by_distance_and_animal is None):
            agg_columns = ['animal', 'environment', binned_measure] + ([] if hue is None else [hue])
            if aggregate_by_distance_and_animal == 'mean':
                df = df.groupby(agg_columns).mean().reset_index()
            elif aggregate_by_distance_and_animal == 'median':
                df = df.groupby(agg_columns).median().reset_index()
            else:
                f = aggregate_by_distance_and_animal
                df = df.groupby(agg_columns).apply(f).reset_index()

        super(ValueByBinnedDistancePlot, self).__init__(df, value, binned_measure, *args,
                                                        hue=hue, hue_order=hue_order, **kwargs)

        if not (data_selection_label_kwargs is None):

            for ax, environment in zip(self._axs[1:], self._environments[1:]):

                ax.set_title('')
                spatial_filter_legend_instance.append_to_axes(
                    ax, experiment_id_substitutes_inverse[environment],
                    distance_measure=binned_measure,
                    distance_bin_width=ValueByBinnedDistancePlot.distance_bin_width,
                    filter_to_middle_third=filter_to_middle_third,
                    filter_to_first_bin_from_wall=filter_to_first_bin_from_wall,
                    **data_selection_label_kwargs
                )

    @staticmethod
    def bin_distance_values(df, distance_column_name, distance_bin_width=None,
                            environment_column='environment'):

        if distance_bin_width is None:
            distance_bin_width = ValueByBinnedDistancePlot.distance_bin_width

        # Bin distance values to ranges specified with distance_bin_width
        distance_bin_edges = np.arange(0, df[distance_column_name].max() + 0.00001,
                                       distance_bin_width)
        distance_bin_centers = distance_bin_edges[:-1] + distance_bin_width / 2.
        df[distance_column_name] = np.array(pd.cut(df[distance_column_name],
                                                   distance_bin_edges, labels=distance_bin_centers))
        df.drop(df.index[np.isnan(df[distance_column_name])], inplace=True)

        cut_df_rows_by_distance_to_wall(df, distance_bin_width, distance_column=distance_column_name,
                                        environment_column=environment_column)
        if distance_bin_width % 2 == 0:
            df[distance_column_name] = df[distance_column_name].astype(np.int16)

    @staticmethod
    def filter_to_middle_third(df):
        return df[df['distance_to_long_wall_prop']
                  >= ValueByBinnedDistancePlot.long_wall_proportional_distance_middle_third_filter]

    @staticmethod
    def filter_to_middle_fifth(df):
        return df[df['distance_to_long_wall_prop']
                  >= ValueByBinnedDistancePlot.long_wall_proportional_distance_middle_fifth_filter]

    @staticmethod
    def filter_to_first_bin_from_wall(df, distance_bin_width=None):
        if distance_bin_width is None:
            distance_bin_width = ValueByBinnedDistancePlot.distance_bin_width
        return df[df['distance to wall (cm)'] < distance_bin_width]


def compute_width_orthogonal_to_wall(ratemap, bin_size, peak_x, peak_y, arena_size, rule='any'):
    """Returns the width of the field in a field ratemap on the axis orthogonal to nearest wall (to peak).
    The width is computed as the length of the projection of the field on the axis othogonal to wall.

    :param numpy.ndarray ratemap: field ratemap
    :param float bin_size: length of ratemap bin side
    :param float peak_x:
    :param float peak_y:
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :param str rule: one of ('any', 'long', 'short') to determine if width is computed:
        parallel to any closest wall
        parallel to closest long wall (of the two axes based on arena size)
        parallel to closest short wall (of the two axes based on arena size)
    :return: width
    :rtype: float
    """

    if rule == 'any':
        axis = snippets.axis_of_nearest_wall((peak_x, peak_y), arena_size)
    elif rule == 'long':
        axis = (1, 0)[int(np.argmax(arena_size))]  # (0, for vertical; 1, for horizontal axis)
    elif rule == 'short':
        axis = (1, 0)[int(np.argmin(arena_size))]  # (0, for vertical; 1, for horizontal axis)
    else:
        raise ValueError('Unexpected argument rule {}'.format(rule))

    return np.sum(np.any(~np.isnan(ratemap), axis=axis)) * bin_size


def compute_width_parallel_to_wall(ratemap, bin_size, peak_x, peak_y, arena_size, rule='any'):
    """Returns the width of the field in a field ratemap on the axis parallel to nearest wall (to peak).
    The width is computed as the length of the projection of the field on the axis parallel to wall.

    :param numpy.ndarray ratemap: field ratemap
    :param float bin_size: length of ratemap bin side
    :param float peak_x:
    :param float peak_y:
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :param str rule: one of ('any', 'long', 'short') to determine if width is computed:
        parallel to any closest wall
        parallel to closest long wall (of the two axes based on arena size)
        parallel to closest short wall (of the two axes based on arena size)
    :return: width
    :rtype: float
    """

    if rule == 'any':
        axis = snippets.axis_of_nearest_wall((peak_x, peak_y), arena_size)
    elif rule == 'long':
        axis = (1, 0)[int(np.argmax(arena_size))]  # (0, for vertical; 1, for horizontal axis)
    elif rule == 'short':
        axis = (1, 0)[int(np.argmin(arena_size))]  # (0, for vertical; 1, for horizontal axis)
    else:
        raise ValueError('Unexpected argument rule {}'.format(rule))

    return np.sum(np.any(~np.isnan(ratemap), axis=(1, 0)[axis])) * bin_size


def directional_ratemap_peak_deviation_environment_axes(bins, rates, arena_size, rule):

    if rule == 'long':
        wall_vector = ((1, 0), (0, 1))[int(np.argmax(arena_size))]
    elif rule == 'short':
        wall_vector = ((1, 0), (0, 1))[int(np.argmin(arena_size))]
    else:
        raise ValueError('Unexpected argument rule {}'.format(rule))

    wall_angle = np.angle(complex(*wall_vector))
    peak_angle = bins[np.nanargmax(rates)]

    return snippets.angular_deviation_from_line_angle(peak_angle, wall_angle)


class RatemapsPlot(object):

    ratemap_plot_labels = {
            'exp_scales_a': 'Env. A',
            'exp_scales_b': 'Env. B',
            'exp_scales_c': 'Env. C',
            'exp_scales_d': 'Env. D',
            'exp_scales_a2': "Env. A'"
        }

    default_figure_size = (6, 8)

    @staticmethod
    def create_spatial_window(arena_size):
        return 0, arena_size[0], 0, arena_size[1]

    @staticmethod
    def ratemap_position_argument(experiment_id, spatial_window):

        margins = 25

        full_height = margins + 125 + margins + 125 + margins + 250 + margins
        full_width = margins + 87.5 + margins + 87.5 + margins + 175 + margins
        h = 1. / full_height
        w = 1. / full_width

        position = {
            'exp_scales_a': (margins * w, (margins + 250 + margins + 125 + margins) * h),
            'exp_scales_b': (margins * w, (margins + 250 + margins) * h),
            'exp_scales_c': ((margins + 87.5 + margins + 87.5 + margins) * w, (margins + 250 + margins) * h),
            'exp_scales_d': (margins * w, margins * h),
            'exp_scales_a2': ((margins + 87.5 + margins) * w, (margins + 250 + margins + 125 + margins) * h)
        }[experiment_id]

        width = (spatial_window[1] - spatial_window[0]) * w
        height = (spatial_window[3] - spatial_window[2]) * h

        position += (width, height)

        return position

    @staticmethod
    def colorbar_position_argument():

        margins = 25

        full_height = margins + 125 + margins + 125 + margins + 250 + margins
        full_width = margins + 87.5 + margins + 87.5 + margins + 175 + margins
        h = 1. / full_height
        w = 1. / full_width

        return ((margins + 350 + margins) * w,
                margins * h,
                10 * w,
                125 * h)

    @staticmethod
    def plot(recordings, recordings_unit_ind, fig, horizontal_offset=0, plain=False, draw_gaussians=True,
             draw_ellipses=True):
        """Plots composite ratemap plot of 5 different recordings on the input figure.

        The figure provided must have the ratio of height to width of 12/9
        or by specifying the left edge of ratemap plot area with horizontal_offset,
        the shape of the remaining area on the figure must have the same ratio.

        For example:
            figure shape (4.5 width, 6 height) and horizontal_offset=0
            figure shape (18, 6) and horizontal_offset=0.75

        :param recordings: :py:class:`recording_io.Recordings`
        :param recordings_unit_ind: unit position in `recordings.units`
        :param matplotlib.figure.Figure fig: figure to plot ratemaps in
        :param float horizontal_offset: percentage (relative to `fig` width) of horizontal
            rightwards shift of the left edge of this composite plot of ratemaps.
        :param bool plain: if True, skips plotting contours, model and ellipse fits. Default is False.
        :param bool draw_gaussians: if True (default), available gaussians are drawn
        """

        horizontal_multiplier = 1 - horizontal_offset

        experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c',
                          'exp_scales_d', 'exp_scales_a2')

        ax = None  # just to keep things neat in PyCharm

        for recording, unit in zip(recordings, recordings.units[recordings_unit_ind]):

            # Get field contours and fitted_models if any for this unit in this recording
            field_inds = np.where((recordings.df_fields['experiment_id'] == recording.info['experiment_id'])
                                  & (recordings.df_fields['animal_unit'] == recordings_unit_ind))[0]
            if len(field_inds) > 0 and not plain:
                contours = [compute_field_contour(recordings[0].analysis['fields'][i]['ratemap']) for i in field_inds]
                if draw_gaussians:
                    fitted_model_params = [{k[6:]: v for k, v in recordings[0].analysis['fields'][i]['properties'].items()
                                            if k.startswith('gauss_')} for i in field_inds]
                    fitted_models = [Gaussian2D(**params) for params in fitted_model_params]
                else:
                    fitted_model_params = None
                    fitted_models = None
                if draw_ellipses:
                    ellipse_params = [recordings[0].analysis['fields'][i]['regionprops_params'] for i in field_inds]
                else:
                    draw_ellipses = None
            else:
                contours = None
                fitted_models = None
                ellipse_params = None

            if not draw_gaussians:
                fitted_models = None

            if not draw_ellipses:
                ellipse_params = None

            if recording.info['experiment_id'] in experiment_ids and not (unit is None):
                i = experiment_ids.index(recording.info['experiment_id'])

                # This should only rely on what is in the unit dictionary
                spatial_window = (RatemapsPlot.create_spatial_window(recording.info['arena_size'])
                                  if not ('arena_size' in unit['analysis']['spatial_ratemaps'])
                                  else unit['analysis']['spatial_ratemaps']['arena_size'])

                ratemap_position = RatemapsPlot.ratemap_position_argument(experiment_ids[i], spatial_window)
                ratemap_position = (ratemap_position[0] * horizontal_multiplier + horizontal_offset,
                                    ratemap_position[1],
                                    ratemap_position[2] * horizontal_multiplier,
                                    ratemap_position[3])
                ax = fig.add_subplot(position=ratemap_position)

                SpatialRatemap.plot(
                    unit['analysis']['spatial_ratemaps']['spike_rates_smoothed'],
                    spatial_window, ax, cmap='jet', contours=contours, fitted_models=fitted_models,
                    ellipse_params=ellipse_params
                )

                max_rate = np.nanmax(unit['analysis']['spatial_ratemaps']['spike_rates_smoothed'].flatten())

                ax.text(1, 1, '{:.1f} Hz'.format(max_rate),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=12)

                if recording.info['experiment_id'] in RatemapsPlot.ratemap_plot_labels:
                    ax.text(0, 1, RatemapsPlot.ratemap_plot_labels[recording.info['experiment_id']],
                            horizontalalignment='left',
                            verticalalignment='bottom',
                            transform=ax.transAxes,
                            fontsize=12)

                ax.axis('off')

        if not (ax is None):
            colorbar_position = RatemapsPlot.colorbar_position_argument()
            colorbar_position = (colorbar_position[0] * horizontal_multiplier + horizontal_offset,
                                 colorbar_position[1],
                                 colorbar_position[2] * horizontal_multiplier,
                                 colorbar_position[3])
            cax = fig.add_subplot(position=colorbar_position)
            fig.colorbar(ax.get_images()[0], cax=cax, ax=ax)
            cax.axis('off')

    @staticmethod
    def plot_other_as_ratemap(data, fig, horizontal_offset=0, cmap='jet', same_colormap_range=True,
                              colormap_range=None, experiment_ids=None):
        """Plots composite ratemap plot of 5 different recordings on the input figure.

        The figure provided must have the ratio of height to width of 12/9
        or by specifying the left edge of ratemap plot area with horizontal_offset,
        the shape of the remaining area on the figure must have the same ratio.

        For example:
            figure shape (4.5 width, 6 height) and horizontal_offset=0
            figure shape (18, 6) and horizontal_offset=0.75

        :param dict data: dictionary with values to plot and keys for each element experiment_ids
        :param matplotlib.figure.Figure fig: figure to plot ratemaps in
        :param float horizontal_offset: percentage (relative to `fig` width) of horizontal
            rightwards shift of the left edge of this composite plot of ratemaps.
        :param str cmap: passed on to :py:func:`SpatialRatemap.plot`
        :param bool same_colormap_range: if True (default), colormap on all plots will have the same range.
        :param tuple colormap_range: (min_value, max_value) for scaling the colormaps
            in conjunction with same_colormap_range=True.
        :param tuple experiment_ids: iterable of strings to match keys in `data` and options for
            :py:attr:`RatemapsPlot.ratemap_position_argument`. If None (default), experiment_ids
            takes the value: `('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d', 'exp_scales_a2')`.
        :return dict of :py:class:`matplotlib.axes._subplots.AxesSubplot` with keys corresponding to data
        :rtype: dict
        """

        horizontal_multiplier = 1 - horizontal_offset

        if experiment_ids is None:
            experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c',
                              'exp_scales_d', 'exp_scales_a2')

        # TODO: This should be inferred somehow from input not defined here
        spatial_windows = {
            'exp_scales_a': (0, 87.5, 0, 125),
            'exp_scales_b': (0, 175, 0, 125),
            'exp_scales_c': (0, 175, 0, 250),
            'exp_scales_d': (0, 350, 0, 250),
            'exp_scales_a2': (0, 87.5, 0, 125)
        }

        ax = None  # just to keep things neat in PyCharm

        if same_colormap_range:
            if colormap_range is None:
                min_value = min([np.nanmin(data[experiment_id].flatten()) for experiment_id in experiment_ids])
                max_value = max([np.nanmax(data[experiment_id].flatten()) for experiment_id in experiment_ids])
            else:
                min_value, max_value = colormap_range
        else:
            min_value, max_value = (None, None)

        axs = {}
        for experiment_id in data:

            i = experiment_ids.index(experiment_id)

            # This should only rely on what is in the unit dictionary
            spatial_window = spatial_windows[experiment_id]

            ratemap_position = RatemapsPlot.ratemap_position_argument(experiment_ids[i], spatial_window)
            ratemap_position = (ratemap_position[0] * horizontal_multiplier + horizontal_offset,
                                ratemap_position[1],
                                ratemap_position[2] * horizontal_multiplier,
                                ratemap_position[3])
            ax = fig.add_subplot(position=ratemap_position)

            if not same_colormap_range:
                with warnings.catch_warnings():
                    warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                            message='All-NaN slice encountered')
                    min_value = np.nanmin(data[experiment_id].flatten())
                    max_value = np.nanmax(data[experiment_id].flatten())

                ax.text(1, 1, '{:.3f} - {:.3f}'.format(min_value, max_value),
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes)

            SpatialRatemap.plot(data[experiment_id], spatial_window, ax, cmap=cmap,
                                vmin=min_value, vmax=max_value)

            if experiment_id in RatemapsPlot.ratemap_plot_labels:
                ax.text(0, 1, RatemapsPlot.ratemap_plot_labels[experiment_id],
                        horizontalalignment='right',
                        verticalalignment='bottom',
                        transform=ax.transAxes,
                        fontsize=14, fontweight='bold')

            ax.axis('off')

            axs[experiment_id] = ax

        if not (ax is None):
            colorbar_position = RatemapsPlot.colorbar_position_argument()
            colorbar_position = (colorbar_position[0] * horizontal_multiplier + horizontal_offset,
                                 colorbar_position[1],
                                 colorbar_position[2] * horizontal_multiplier,
                                 colorbar_position[3])
            cax = fig.add_subplot(position=colorbar_position)
            fig.colorbar(ax.get_images()[0], cax=cax, ax=ax)
            cax.axis('off')

            if same_colormap_range:

                cax.text(1, 1, '{:.3f}'.format(max_value),
                         horizontalalignment='center',
                         verticalalignment='bottom',
                         transform=cax.transAxes)
                cax.text(1, 0, '{:.3f}'.format(min_value),
                         horizontalalignment='center',
                         verticalalignment='top',
                         transform=cax.transAxes)

        return axs

    @staticmethod
    def make_default_figure():
        return plt.figure(figsize=RatemapsPlot.default_figure_size)


class PlaceFieldPropertyMaps(object):

    properties_to_plot = ('count', 'stability_minutes', 'stability_halves', 'peak_spike_rate',
                          'local_spike_rate_per_unit', 'median_spike_rate', 'peak_x', 'peak_y', 'area', 'axes_ratio',
                          'major_axis_theta', 'centroid_x', 'centroid_y', 'eccentricity', 'minor_axis', 'major_axis',
                          'orientation', 'peak_nearest_corner', 'centroid_nearest_corner', 'peak_nearest_wall',
                          'centroid_nearest_wall', 'peak_and_shortest_stddev', 'peak_and_radius', 'area_radius',
                          'firing_rate_gradient', 'fisher_information', 'width_orthogonal_to_wall',
                          'width_parallel_to_wall', 'directional_firing_mean_rv',
                          'directional_kl_divergence', 'directional_kl_divergence_js', 'directional_kl_divergence_sym',
                          'width_parallel_to_long_wall', 'width_orthogonal_to_long_wall',
                          'width_parallel_to_short_wall', 'width_orthogonal_to_short_wall')

    property_plot_names = {
        'count': 'fields per unit',
        'peak_spike_rate': 'peak firing rate (Hz)',
        'local_spike_rate_per_unit': 'local firing rate per unit (Hz)',
        'peak_and_radius': 'peak to radius ratio',
        'area_radius': 'field radius',
        'firing_rate_gradient': 'firing rate gradient per unit (Hz)',
        'ratemap_fisher_information': 'fisher information per unit',
        'width_orthogonal_to_wall': 'field width orthogonal to wall',
        'width_parallel_to_wall': 'field width parallel to wall'
    }

    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c',
                      'exp_scales_d', 'exp_scales_a2')

    spatial_windows = {
        'exp_scales_a': (0, 87.5, 0, 125),
        'exp_scales_b': (0, 175, 0, 125),
        'exp_scales_c': (0, 175, 0, 250),
        'exp_scales_d': (0, 350, 0, 250),
        'exp_scales_a2': (0, 87.5, 0, 125)
    }

    def __init__(self, all_recordings, df_units, df_fields, unit_category='place_cell',
                 property_name=None, verbose=False):
        """

        :param all_recordings:
        :param df_units:
        :param df_fields: df_fields must contain column 'unit' that links to each row in df_units
        :param property_name: allows limiting computation to a single property per instance of PlaceFieldPropertyMaps
        :param verbose:
        """

        self.all_recordings = all_recordings
        self.df_units = df_units
        self.df_fields = df_fields
        self.unit_category = unit_category
        self.property_name = property_name
        self.verbose = verbose

        self.field_property_maps = {}
        self.unit_counts = {}
        self.compute_property_maps()

    def compute_property_maps(self):

        # Collect into dictionary maps for each property in each experiment_id

        if self.verbose:
            print('Computing field_property_maps')

        properties_to_compute = self.properties_to_plot if self.property_name is None else [self.property_name]

        for property_name in properties_to_compute:

            self.field_property_maps[property_name] = {experiment_id: [] for experiment_id in self.experiment_ids}

            for experiment_id in self.experiment_ids:
                self.field_property_maps[property_name][experiment_id] = {animal: [] for animal in Params.animal_ids}

        # Compute separately for each experiment_id
        for experiment_id in self.experiment_ids:

            # Iterate over all place cells
            for unit_ind in np.where(self.df_units['category'] == self.unit_category)[0]:

                # Get the df_fields for this unit and experiment_id
                df_unit_experiment_fields = self.df_fields[(self.df_fields['unit'] == unit_ind)
                                                           & (self.df_fields['experiment_id'] == experiment_id)]

                # Iterate over all fields that matched this unit and experiment_id
                for _, df_field in df_unit_experiment_fields.iterrows():

                    # Get the ratemap for this field
                    animal_recordings = snippets.get_first_animal_recordings(self.all_recordings, df_field['animal'])
                    field_ratemap = animal_recordings[0].analysis['fields'][df_field['animal_field']]['ratemap']

                    # Extract the values for each property
                    for property_name in properties_to_compute:

                        # Create a map where values inside this field are the property values, but np.nan outside
                        property_map = np.float32(field_ratemap.copy())
                        if property_name == 'count':
                            property_map = np.float32(~np.isnan(property_map))
                        elif property_name == 'area_radius':
                            property_map[~np.isnan(property_map)] = \
                                np.sqrt(np.float32(np.float32(df_field['area']) / np.pi))
                        elif property_name == 'local_spike_rate_per_unit':
                            pass
                        elif property_name == 'firing_rate_gradient':
                            property_map = ratemap_gradient(property_map)
                        elif property_name == 'fisher_information':
                            property_map = ratemap_fisher_information(property_map)
                        elif property_name == 'width_parallel_to_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_parallel_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                               df_field['peak_x'], df_field['peak_y'],
                                                               (spatial_windows[df_field['experiment_id']][1],
                                                                spatial_windows[df_field['experiment_id']][3]))
                        elif property_name == 'width_orthogonal_to_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_orthogonal_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                                 df_field['peak_x'], df_field['peak_y'],
                                                                 (spatial_windows[df_field['experiment_id']][1],
                                                                  spatial_windows[df_field['experiment_id']][3]))
                        elif property_name == 'width_parallel_to_long_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_parallel_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                               df_field['peak_x'], df_field['peak_y'],
                                                               (spatial_windows[df_field['experiment_id']][1],
                                                                spatial_windows[df_field['experiment_id']][3]),
                                                               rule='long')
                        elif property_name == 'width_orthogonal_to_long_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_orthogonal_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                                 df_field['peak_x'], df_field['peak_y'],
                                                                 (spatial_windows[df_field['experiment_id']][1],
                                                                  spatial_windows[df_field['experiment_id']][3]),
                                                                 rule='long')
                        elif property_name == 'width_parallel_to_short_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_parallel_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                               df_field['peak_x'], df_field['peak_y'],
                                                               (spatial_windows[df_field['experiment_id']][1],
                                                                spatial_windows[df_field['experiment_id']][3]),
                                                               rule='short')
                        elif property_name == 'width_orthogonal_to_short_wall':
                            property_map[~np.isnan(property_map)] = \
                                compute_width_orthogonal_to_wall(property_map, Params.spatial_ratemap['bin_size'],
                                                                 df_field['peak_x'], df_field['peak_y'],
                                                                 (spatial_windows[df_field['experiment_id']][1],
                                                                  spatial_windows[df_field['experiment_id']][3]),
                                                                 rule='short')
                        elif property_name == 'preferred_direction_angle_to_short_wall':
                            field_dict = animal_recordings[0].analysis['fields'][df_field['animal_field']]
                            property_map[~np.isnan(property_map)] = \
                                directional_ratemap_peak_deviation_environment_axes(
                                    field_dict['directional_ratemap']['bins'],
                                    field_dict['directional_ratemap']['spike_rate_smoothed'],
                                    (spatial_windows[df_field['experiment_id']][1],
                                     spatial_windows[df_field['experiment_id']][3]),
                                    'short'
                                )
                        elif property_name == 'preferred_direction_angle_to_long_wall':
                            field_dict = animal_recordings[0].analysis['fields'][df_field['animal_field']]
                            property_map[~np.isnan(property_map)] = \
                                directional_ratemap_peak_deviation_environment_axes(
                                    field_dict['directional_ratemap']['bins'],
                                    field_dict['directional_ratemap']['spike_rate_smoothed'],
                                    (spatial_windows[df_field['experiment_id']][1],
                                     spatial_windows[df_field['experiment_id']][3]),
                                    'long'
                                )
                        else:
                            property_map[~np.isnan(property_map)] = np.float32(df_field[property_name])

                        # Append this field property map to correct position in the field_property_maps dictionary
                        self.field_property_maps[property_name][experiment_id][df_field['animal']].append(property_map)

        # Convert all field lists into arrays per animal and mask nans
        for property_name in properties_to_compute:

            for experiment_id in self.experiment_ids:

                for animal in self.field_property_maps[property_name][experiment_id]:

                    self.field_property_maps[property_name][experiment_id][animal] = \
                        np.stack(self.field_property_maps[property_name][experiment_id][animal], axis=2)

                    self.field_property_maps[property_name][experiment_id][animal] = \
                        np.ma.array(self.field_property_maps[property_name][experiment_id][animal],
                                    mask=np.isnan(self.field_property_maps[property_name][experiment_id][animal]))

    def get_maps_for_animal(self, property_name, experiment_id, animal, min_n_fields_per_bin=0):

        # Get the property mask
        property_maps = self.field_property_maps[property_name][experiment_id][animal]

        if min_n_fields_per_bin > 0:

            # Find bins to exclude based on number of overlapping the bin
            field_count_array = self.field_property_maps['count'][experiment_id][animal]
            bins_to_exclude = np.sum(field_count_array, axis=2) < min_n_fields_per_bin

            # Update mask with the bin sampling exclusion array
            bins_to_exclude = np.stack([bins_to_exclude] * property_maps.shape[2], axis=2)
            property_maps.mask = property_maps.mask | bins_to_exclude

        return property_maps

    def get_maps_across_animals(self, property_name, experiment_id, min_n_fields_per_bin=0):

        # Collect arrays per animal
        property_maps = []
        field_count_array = []
        for animal in Params.animal_ids:

            property_maps.append(self.get_maps_for_animal(property_name, experiment_id, animal))

            if min_n_fields_per_bin > 0:
                field_count_array.append(self.get_maps_for_animal('count', experiment_id, animal))

        # Concatenate maps array from all animals
        property_maps = np.ma.concatenate(property_maps, axis=2)

        if min_n_fields_per_bin > 0:

            # Concatenate field count array from all animals
            field_count_array = np.ma.concatenate(field_count_array, axis=2)

            # Find bins to exclude based on number of overlapping the bin
            bins_to_exclude = np.sum(field_count_array, axis=2) < min_n_fields_per_bin

            # Update mask with the bin sampling exclusion array
            bins_to_exclude = np.stack([bins_to_exclude] * property_maps.shape[2], axis=2)
            property_maps.mask = property_maps.mask | bins_to_exclude

        return property_maps

    def get_property_summary_array(self, property_name, experiment_id, animal=None,
                                   min_n_fields_per_bin=0):

        if animal is None:
            property_maps = self.get_maps_across_animals(property_name, experiment_id,
                                                         min_n_fields_per_bin=min_n_fields_per_bin)
        else:
            property_maps = self.get_maps_for_animal(property_name, experiment_id, animal,
                                                     min_n_fields_per_bin=min_n_fields_per_bin)

        # n_units is used by some versions of func below
        if animal is None:
            n_units = float(sum(self.df_units['category'] == self.unit_category))
        else:
            n_units = float(sum((self.df_units['category'] == self.unit_category)
                                & (self.df_units['animal'] == animal)))

        if property_name in ['count', 'local_spike_rate_per_unit', 'firing_rate_gradient',
                             'fisher_information']:
            def func(x):
                return np.ma.divide(np.ma.sum(x, axis=2), n_units)
        elif property_name == 'major_axis_theta':
            def func(x):
                return np.ma.apply_along_axis(lambda xi: circmean(xi.data[~xi.mask], high=np.pi, low=0),
                                              2, x)
        elif property_name == 'orientation':
            def func(x):
                return np.ma.apply_along_axis(lambda xi: circmean(xi.data[~xi.mask], high=np.pi, low=0),
                                              2, x)
        else:
            def func(x):
                return np.ma.median(x, axis=2)

        property_map = func(property_maps)

        # Set masked bins to numpy.nan and keep a simple array
        if isinstance(property_map, np.ma.core.MaskedArray):
            property_map = np.array(property_map.filled(np.nan).data)

        return property_map

    def plot_maps_as_arrays_combining_animals(self, fpath, property_name, min_n_fields_per_bin=0):

        sns.set_context('notebook')

        property_maps = {}
        for experiment_id in self.experiment_ids:
            property_maps[experiment_id] = self.get_property_summary_array(
                property_name, experiment_id, min_n_fields_per_bin=min_n_fields_per_bin
            )

        for colormap_method, same_colormap_range in zip(['same_colormap', 'independent_colormap'], [True, False]):

            fig = RatemapsPlot.make_default_figure()

            RatemapsPlot.plot_other_as_ratemap(
                property_maps, fig,
                cmap=('twilight' if property_name in ['major_axis_theta', 'orientation'] else 'jet'),
                same_colormap_range=same_colormap_range
            )

            fig.savefig(os.path.join(fpath, '{}_map_{}.png'.format(property_name, colormap_method)))

            plt.close()

    def plot_map_data_as_lineplots_from_boundary(self, fpath, property_name, min_n_fields_per_bin=0,
                                                 distance_bin_width=10,
                                                 skip_exp_scales_a=False, count_plot_bins_any=True):

        if count_plot_bins_any and property_name == 'count':
            min_n_fields_per_bin = 0

        if skip_exp_scales_a:
            experiment_ids = self.experiment_ids[1:-1]
        else:
            experiment_ids = self.experiment_ids[:-1]

        # Plot field property distributions relative to wall

        sns.set(color_codes=True)
        sns.set_context('talk', rc={'lines.linewidth': 5})

        bin_distance_to_wall = snippets.SpatialMapBinCenterDistanceToWall()

        dfs = []

        for animal in Params.animal_ids:

            for experiment_id in experiment_ids:

                property_map = self.get_property_summary_array(
                    property_name, experiment_id, animal=animal, min_n_fields_per_bin=min_n_fields_per_bin
                )

                distances = []
                values = []
                n_xbins = []
                n_ybins = []
                for n_xbin in range(property_map.shape[1]):
                    for n_ybin in range(property_map.shape[0]):
                        distances.append(bin_distance_to_wall.get(property_map.shape,
                                                                  self.spatial_windows[experiment_id],
                                                                  n_xbin, n_ybin))
                        values.append(property_map[n_ybin, n_xbin])
                        n_xbins.append(n_xbin)
                        n_ybins.append(n_ybin)

                distance_from_corner = snippets.compute_distance_to_nearest_corner_for_array(
                    np.stack((n_xbins, n_ybins), axis=1).astype(np.float32) + 0.5,
                    np.array(property_map.shape[:2][::-1])
                )
                distance_from_corner = distance_from_corner / (np.min(property_map.shape[:2]) / 2.)

                dfs.append(pd.DataFrame(
                    {
                        'distance': distances,
                        'value': values,
                        'experiment_id': [experiment_id_substitutes[experiment_id]] * len(values),
                        'animal': [animal] * len(values),
                        'proportional_distance_from_corner': distance_from_corner
                    }
                ))

        df_property_values = pd.concat(dfs, ignore_index=True)

        # Drop any rows containing NaN values
        df_property_values.dropna(inplace=True)

        # Bin distance values to ranges specified with distance_bin_width
        distance_bin_edges = np.arange(0, df_property_values['distance'].max(), distance_bin_width)
        distance_bin_centers = distance_bin_edges[:-1] + distance_bin_width / 2.
        df_property_values['distance'] = pd.cut(df_property_values['distance'],
                                                distance_bin_edges,
                                                labels=distance_bin_centers)

        cut_df_rows_by_distance_to_wall(df_property_values, distance_bin_width,
                                        environment_column='experiment_id',  distance_column='distance')

        # Plot seaprately with different rules for inclusion based on distance to corner

        for corner_rule in (None, 'ignore', 'only'):

            df = df_property_values.copy(deep=True)

            # Filter for samples that are sufficiently far from corners (if specified)
            if not (corner_rule is None):
                if corner_rule == 'ignore':
                    df = df[df['proportional_distance_from_corner'] >= 1].copy().reset_index()
                elif corner_rule == 'only':
                    df = df[df['proportional_distance_from_corner'] < 1].copy().reset_index()
                else:
                    raise ValueError(
                        'Unknown corner rule {}, only None, "ignore" or "only" allowed'.format(corner_rule))

            # Compute median values across all position bins in a distance bin

            df = df.groupby(['distance', 'experiment_id', 'animal'])
            if property_name in ['major_axis_theta', 'orientation']:
                df = df.agg(circmean, high=np.pi, low=0).reset_index()
            else:
                df = df.agg(np.mean).reset_index()

            if property_name in self.property_plot_names:
                property_name_in_plot = self.property_plot_names[property_name]
            else:
                property_name_in_plot = property_name

            df.rename(columns={'value': property_name_in_plot,
                               'experiment_id': 'environment',
                               'distance': 'distance to wall'},
                      inplace=True)

            plt.figure(figsize=(9, 6))

            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                        message='Mean of empty slice.')
                warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                        message='invalid value encountered in double_scalars')
                sns.lineplot(x='distance to wall', y=property_name_in_plot, hue='environment', data=df,
                             sizes=(5, 5))

            plt.tight_layout(pad=3)

            plt.savefig(os.path.join(fpath, '{}_corner_rule={}_lineplot.png'.format(property_name, corner_rule)))

            plt.close()

    def plot_all(self, fpath, combined_map_kwargs, lineplot_kwargs):

        plots_path = os.path.join(fpath, Params.analysis_path, 'FieldPropertyMaps')

        # Make sure path is clear to save figures
        if os.path.isdir(plots_path):
            rmtree(plots_path)
        os.mkdir(plots_path)

        for property_name in self.properties_to_plot:

            self.plot_maps_as_arrays_combining_animals(plots_path, property_name, **combined_map_kwargs)
            self.plot_map_data_as_lineplots_from_boundary(plots_path, property_name, **lineplot_kwargs)


class RatemapOverlapRateGradientAndFisherInformationMaps(object):

    @staticmethod
    def get_recording_ratemaps(recordings, i_recording, recompute=False, ratemap_kwargs=None):
        """Returns a stack of ratemaps for all place cells and includes all zero ratemaps
        for units that don't have a single spike in a given recording.

        :param recordings:
        :param i_recording:
        :return:
        """
        ratemaps = []
        ratemap_shape = None
        for i, unit in enumerate(recordings.units):
            if recordings.first_available_recording_unit(i)['analysis']['category'] != 'place_cell':
                continue
            if unit[i_recording] is None:
                ratemaps.append(None)
            else:

                if recompute:

                    ratemap = SpatialRatemap(
                        recordings[i_recording].position['xy'],
                        unit[i_recording]['timestamps'],
                        recordings[i_recording].position['sampling_rate'], ratemap_kwargs['bin_size'],
                        spatial_window=spatial_windows[recordings[i_recording].info['experiment_id']],
                        xy_mask=recordings[i_recording].position['analysis']['ratemap_speed_mask'],
                        n_samples=(ratemap_kwargs['n_samples'] if 'n_samples' in ratemap_kwargs else None)
                    )
                    ratemaps.append(
                        ratemap.spike_rates_smoothed(n_bins=ratemap_kwargs['n_smoothing_bins'],
                                                     method=ratemap_kwargs['smoothing_method'])
                    )
                    ratemap_shape = ratemaps[-1].shape

                else:

                    ratemaps.append(unit[i_recording]['analysis']['spatial_ratemaps']['spike_rates_smoothed'])
                    ratemap_shape = ratemaps[-1].shape

        for i, ratemap in enumerate(ratemaps):
            if ratemap is None:
                ratemaps[i] = np.zeros(ratemap_shape, dtype=np.float64)

        return np.stack(ratemaps, axis=2)

    @staticmethod
    def set_nans_by_min_surrounding_valid_bins(data, valid_bin_idx, min_valid_surrounding_bins):
        kernel = np.ones((3, 3))
        valid_bin_count = convolve2d(valid_bin_idx, kernel, mode='same', boundary='fill', fillvalue=0) - 1
        idx_set_to_nan = valid_bin_count < min_valid_surrounding_bins
        if data.ndim == 2:
            pass
        elif data.ndim == 3:
            idx_set_to_nan = np.stack([idx_set_to_nan] * data.shape[2], axis=2)
        else:
            raise ValueError('Unexpected data shape {}'.format(data.shape))
        data[idx_set_to_nan] = np.nan

    @staticmethod
    def compute_maps(all_recordings, fi_min_rate=1., recompute_ratemaps=False, ratemap_kwargs=None,
                     min_valid_surrounding_bins=0, mean_per_field=False, mean_rate_min_rate=None, verbose=False):

        maps = {method: [] for method in ('active_unit_count', 'firing_rate', 'firing_rate_gradient',
                                          'fisher_information',
                                          'fisher_information_parallel_to_long_wall',
                                          'fisher_information_parallel_to_short_wall')}

        # Get data for all animals
        animal_ids = []
        for recordings in all_recordings:

            animal_ids.append(recordings[0].info['animal'])

            animal_maps = {method: {} for method in maps}

            for i_recording in range(4):

                experiment_id = recordings[i_recording].info['experiment_id']

                if verbose:
                    print('Computing ratemap measures for animal {} experiment {}'.format(
                        recordings[0].info['animal'], experiment_id
                    ))

                ratemaps = RatemapOverlapRateGradientAndFisherInformationMaps.get_recording_ratemaps(
                    recordings, i_recording, recompute=recompute_ratemaps, ratemap_kwargs=ratemap_kwargs
                )

                if mean_per_field:

                    df = recordings.df_fields[['animal', 'animal_unit', 'experiment_id']].copy(deep=True)
                    df = df.merge(recordings.df_units[['animal', 'animal_unit', 'category']].copy(deep=True),
                                  how='left', on=['animal', 'animal_unit'])
                    total = np.sum((df['category'] == 'place_cell') & (df['experiment_id'] == experiment_id))

                else:

                    total = ratemaps.shape[2]

                if min_valid_surrounding_bins > 0:
                    RatemapOverlapRateGradientAndFisherInformationMaps.set_nans_by_min_surrounding_valid_bins(
                        ratemaps, ~np.isnan(ratemaps[:, :, 0]), min_valid_surrounding_bins
                    )

                # The following methods must account for NaNs in the mean

                animal_maps['active_unit_count'][experiment_id] = np.sum(ratemaps > 1, axis=2) / total

                if mean_rate_min_rate is None:
                    animal_maps['firing_rate'][experiment_id] = np.nansum(ratemaps, axis=2) / total
                else:
                    idx_active = ratemaps > mean_rate_min_rate
                    masked_map = np.ma.masked_array(ratemaps, mask=~idx_active)
                    animal_maps['firing_rate'][experiment_id] = \
                        np.nansum(masked_map, axis=2).data / np.sum(idx_active, axis=2)

                animal_maps['firing_rate_gradient'][experiment_id] = \
                    np.nansum(ratemap_gradient(ratemaps), axis=2) / total

                animal_maps['fisher_information'][experiment_id] = \
                    (np.nansum(ratemap_fisher_information(ratemaps, min_rate=fi_min_rate), axis=2)
                     / total)

                axis = {'exp_scales_a': 'y', 'exp_scales_b': 'x',
                        'exp_scales_c': 'y', 'exp_scales_d': 'x'}[recordings[i_recording].info['experiment_id']]
                animal_maps['fisher_information_parallel_to_long_wall'][experiment_id] = \
                    (np.nansum(ratemap_fisher_information(ratemaps, min_rate=fi_min_rate, axis=axis), axis=2)
                     / total)

                axis = {'exp_scales_a': 'x', 'exp_scales_b': 'y',
                        'exp_scales_c': 'x', 'exp_scales_d': 'y'}[recordings[i_recording].info['experiment_id']]
                animal_maps['fisher_information_parallel_to_short_wall'][experiment_id] = \
                    (np.nansum(ratemap_fisher_information(ratemaps, min_rate=fi_min_rate, axis=axis), axis=2)
                     / total)

            for method in maps:
                maps[method].append(animal_maps[method])

        return maps, animal_ids


class FieldCoverage(object):

    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def get_mean_ratemap_measure_for_each_position(all_recordings, measure_name, mean_per_field=False,
                                                   mean_rate_min_rate=None):

        maps_by_animal, animal_ids = RatemapOverlapRateGradientAndFisherInformationMaps.compute_maps(
            all_recordings, fi_min_rate=1., recompute_ratemaps=False, mean_per_field=mean_per_field,
            mean_rate_min_rate=mean_rate_min_rate
        )
        maps_by_animal = maps_by_animal[measure_name]

        maps_by_environment = {}
        for experiment_id in FieldCoverage.experiment_ids:

            maps_by_environment[experiment_id] = np.stack([x[experiment_id] for x in maps_by_animal], axis=2)

        df = maps_by_environment_to_df_with_distances(
            maps_by_environment, bin_size=Params.spatial_ratemap['bin_size'], drop_nans=True,
            stack_group_name='animal', stack_groups=animal_ids
        )

        return df


class BayesianPositionDecodingArenaAccuracy(object):

    @staticmethod
    def compute_for_recordings(recordings, position_decoding_name):

        experiment_ids = (
            'exp_scales_a',
            'exp_scales_b',
            'exp_scales_c',
            'exp_scales_d'
        )

        environment_names = [experiment_id_substitutes[recording.info['experiment_id']]
                             for recording in recordings]

        real_environment = []
        decoded_environment = []
        posterior_peaks = []
        posterior_peaks_norm = []
        for recording in [recording for recording in recordings
                          if recording.info['experiment_id'] in experiment_ids]:

            posterior_peaks.append(recording.analysis[position_decoding_name]['peak_value'])
            posterior_peaks_norm.append(recording.analysis[position_decoding_name]['peak_value_normalised'])
            decoded_environment_inds = recording.analysis[position_decoding_name]['peak_recording_ind']
            real_environment += \
                [experiment_id_substitutes[recording.info['experiment_id']]] * len(decoded_environment_inds)
            decoded_environment += [environment_names[i] for i in decoded_environment_inds]

        df = pd.DataFrame({'real environment': real_environment, 'decoded environment': decoded_environment})
        df['percentage'] = np.ones(df.shape[0])
        df = df.groupby(['real environment', 'decoded environment']).sum()
        df = df.divide(df.sum(level='real environment'), level='real environment') * 100

        df_peaks = pd.DataFrame({'real environment': real_environment, 'decoded environment': decoded_environment,
                                 'posterior log peak': np.log(np.concatenate(posterior_peaks)),
                                 'posterior peak (normalised)': np.concatenate(posterior_peaks_norm)})
        df_peaks['group'] = ''
        df_peaks.loc[df_peaks['real environment'] == df_peaks['decoded environment'], 'group'] = 'correct'
        df_peaks.loc[df_peaks['real environment'] != df_peaks['decoded environment'], 'group'] = 'incorrect'
        del df_peaks['decoded environment']
        df_peaks.rename(columns={'real environment': 'environment',
                                 'group': 'decoded environment'}, inplace=True)

        df['animal'] = recordings[0].info['animal']
        df_peaks['animal'] = recordings[0].info['animal']

        return df, df_peaks

    @staticmethod
    def compute_for_all_recordings(all_recordings, position_decoding_name):
        """Returns decoding environment distribution DataFrames for each experiment_id (A, B, C, D) for each
        Recordings instance in all_recordings list and posterior peak values marked for correct and incorrect results

        :param list all_recordings:
        :return: dfs (list of DataFrames), dfs_peaks (list of DataFrames)
        """
        return list(zip(*[
            BayesianPositionDecodingArenaAccuracy.compute_for_recordings(recordings, position_decoding_name)
            for recordings in all_recordings
        ]))

    @staticmethod
    def plot(fpath, all_recordings, position_decoding_names):

        sns.set()
        sns.set_context('paper')

        fig, axs = plt.subplots(1, len(position_decoding_names), figsize=(6 * len(position_decoding_names), 5))

        for position_decoding_name, ax in zip(position_decoding_names, axs):

            dfs, _ = BayesianPositionDecodingArenaAccuracy.compute_for_all_recordings(all_recordings,
                                                                                      position_decoding_name)
            df = pd.concat(dfs).reset_index()

            ax = sns.barplot(data=df, x='real environment', y='percentage', hue='decoded environment', ax=ax)
            ax.set_yscale('log')
            ax.legend(ncol=len(df['decoded environment'].unique()), loc='lower right')
            ax.set_title(position_decoding_name)

        plt.savefig(os.path.join(fpath, Params.analysis_path, 'DecodingEnvironmentAccuracy.png'))
        plt.close(fig)


def filter_dataframe_by_direction(df, rule, section_width='quadrants'):

    idx = np.zeros(df.shape[0], dtype=np.bool)

    direction = np.abs(df['direction'])
    if section_width == 'quadrants':
        vertical_direction = (direction > np.pi / 4) & (direction <= 3 * np.pi / 4)
        horizontal_direction = ~vertical_direction
    elif section_width == 'eights':
        vertical_direction = (direction > 3 * np.pi / 8) & (direction <= 5 * np.pi / 8)
        horizontal_direction = (direction > 7 * np.pi / 8) | (direction <= 1 * np.pi / 8)
    else:
        raise ValueError('Unknown section_width {}'.format(section_width))

    for environment in ('A', 'B', 'C', 'D'):

        idx_environment = df['environment'] == environment

        if environment in ('A', 'C'):

            if rule == 'alonglongwall':
                idx[idx_environment & vertical_direction] = True
            elif rule == 'alongshortwall':
                idx[idx_environment & horizontal_direction] = True
            else:
                raise ValueError

        elif environment in ('B', 'D'):

            if rule == 'alonglongwall':
                idx[idx_environment & horizontal_direction] = True
            elif rule == 'alongshortwall':
                idx[idx_environment & vertical_direction] = True
            else:
                raise ValueError

        else:
            raise ValueError

    return df[idx]


class BayesianPositionDecodingPosteriorProperties(object):

    @staticmethod
    def compute_property_map_for_position_decoding_set(recording, i_recording, position_decoding,
                                                       position_shift, property_name, func, min_bin_count,
                                                       speed_filter_kwargs=None):

        # Get position indices and include a shift if specified
        position_inds = position_decoding['position_inds'] + position_shift
        n_too_early_samples = np.sum(position_inds < 0)
        n_too_late_samples = np.sum(position_inds >= recording.position['xy'].shape[0])

        # Only use samples that were in the correct environment
        if i_recording is None:
            idx_keep = np.ones(position_decoding['position_inds'].shape, dtype=np.bool)
        else:
            idx_keep = position_decoding['peak_recording_ind'] == i_recording

        # Drop too early or too late samples for this position shift
        if n_too_early_samples > 0:
            idx_keep[:n_too_early_samples] = False
            position_inds[:n_too_early_samples] = 0
        if n_too_late_samples > 0:
            idx_keep[-n_too_late_samples:] = False
            position_inds[-n_too_late_samples:] = recording.position['xy'].shape[0] - 1

        # Filter positions outside spatial window
        spatial_window = RatemapsPlot.create_spatial_window(recording.info['arena_size'])
        unfiltered_position_xy = \
            recording.position['xy'][position_inds, :]
        idx_keep = idx_keep & ((unfiltered_position_xy[:, 0] > 0)
                               & (unfiltered_position_xy[:, 0] < spatial_window[1])
                               & (unfiltered_position_xy[:, 1] > 0)
                               & (unfiltered_position_xy[:, 1] < spatial_window[3]))

        # Filter positions based on movement speed
        if not (speed_filter_kwargs is None):
            speed = recording.get_smoothed_speed(speed_filter_kwargs['smoothing_window_length'],
                                                 method=speed_filter_kwargs['smoothing_method'])
            speed_inds = np.where((speed >= speed_filter_kwargs['min_speed'])
                                  & (speed < speed_filter_kwargs['max_speed']))[0]
            idx_keep = idx_keep & np.isin(position_inds, speed_inds, assume_unique=False)

        position_inds = position_inds[idx_keep]

        position_xy_inds = SpatialRatemap.compute_bin_of_xy_after_shifting(
            recording.position['xy'][position_inds, :],
            spatial_window=RatemapsPlot.create_spatial_window(recording.info['arena_size']),
            binsize=Params.bayes_position_decoding['ratemap_kwargs']['bin_size']
        )

        # Compute correct property values
        property_values = None
        if property_name == 'peak_distance':
            property_values = euclidean_distance_between_rows_of_matrix(
                recording.position['xy'][position_inds, :],
                position_decoding['peak_xy'][idx_keep]
            )
        elif property_name == 'log_peak_value':
            property_values = position_decoding['peak_value'][idx_keep]
        elif property_name == 'peak_value_normalised':
            property_values = position_decoding['peak_value_normalised'][idx_keep]

        # Compute counts and mean or median values for each bin
        df = pd.DataFrame({'x_ind': position_xy_inds[:, 0],
                           'y_ind': position_xy_inds[:, 1],
                           'values': property_values})
        property_counts = df.groupby(['x_ind', 'y_ind']).count().reset_index().to_numpy()
        if func == 'mean':
            property_values = df.groupby(['x_ind', 'y_ind']).mean().reset_index().to_numpy()
        elif func == 'median':
            property_values = df.groupby(['x_ind', 'y_ind']).median().reset_index().to_numpy()
        else:
            raise ValueError('expected mean or median but got ' + str(func))
        if property_name == 'log_peak_value':
            property_values[:, 2] = np.log(property_values[:, 2])

        # Filter property values by count
        property_values = property_values[property_counts[:, 2] >= min_bin_count, :]

        # Create the property map
        if 'posterior_shapes' in position_decoding:
            if i_recording is None:
                map_shape = position_decoding['posterior_shape']
            else:
                map_shape = position_decoding['posterior_shapes'][i_recording, :]
        else:
            map_shape = SpatialRatemap.compute_ratemap_shape(
                RatemapsPlot.create_spatial_window(recording.info['arena_size']),
                Params.bayes_position_decoding['ratemap_kwargs']['bin_size']
            )
        property_map = np.zeros(map_shape, dtype=np.float64) * np.nan
        ind_1, ind_2 = (property_values[:, 1].astype(np.int16), property_values[:, 0].astype(np.int16))
        property_map[ind_1, ind_2] = property_values[:, 2]

        return property_map

    @staticmethod
    def compute_property_maps(all_recordings, position_decoding_name, min_bin_count, func, property_name,
                              position_shift=0, position_decoding_var_param=None, subsampling_n_samples=None,
                              speed_filter_kwargs=None):

        # Get similarity data for all animals
        property_maps = []
        for recordings in all_recordings:

            animal_similarity_maps = {}
            for i_recording, recording in enumerate(recordings[:4]):

                position_decoding = None
                position_decoding_sets = None

                if position_decoding_var_param is None and subsampling_n_samples is None:

                    position_decoding = recording.analysis[position_decoding_name]

                elif (not (position_decoding_var_param is None)
                      and position_decoding_name == 'position_decoding_var_param'):

                    position_decoding = None
                    for position_decoding_candidate in recording.analysis[position_decoding_name]:
                        if (
                                (position_decoding_candidate['parameters']['decoding_window_size']
                                 == position_decoding_var_param['decoding_window_size'])
                                and (len(position_decoding_candidate['parameters']['unit_indices'])
                                     == position_decoding_var_param['unit_counts'][recording.info['experiment_id']])
                        ):
                            position_decoding = position_decoding_candidate['results']
                            position_decoding['position_inds'] = position_decoding_candidate['position_inds']
                            break
                    assert not (position_decoding is None), 'If False, parameters not matched in data'

                elif not (subsampling_n_samples is None):

                    position_decoding_sets = []

                    for position_decoding_candidate in recording.analysis['position_decoding_subsampled']:

                        if (
                                position_decoding_candidate['parameters']['name'] == position_decoding_name
                                and (position_decoding_candidate['parameters']['ratemap_kwargs']['n_samples']
                                     == subsampling_n_samples)
                        ):
                            position_decoding = position_decoding_candidate['results']
                            position_decoding['position_inds'] = position_decoding_candidate['position_inds']

                            position_decoding_sets.append(position_decoding)

                    assert len(position_decoding_sets) > 0, ('No examples for position_decoding_subsampled ' +
                                                             'n_samples {} not found'.format(subsampling_n_samples))

                else:

                    raise ValueError('Incorrect inputs.')

                if subsampling_n_samples is None:

                    animal_similarity_maps[recording.info['experiment_id']] = \
                        BayesianPositionDecodingPosteriorProperties.compute_property_map_for_position_decoding_set(
                            recording, i_recording, position_decoding,
                            position_shift, property_name, func, min_bin_count,
                            speed_filter_kwargs=speed_filter_kwargs
                        )

                else:

                    animal_similarity_maps[recording.info['experiment_id']] = []

                    for position_decoding in position_decoding_sets:

                        animal_similarity_maps[recording.info['experiment_id']].append(
                            BayesianPositionDecodingPosteriorProperties.compute_property_map_for_position_decoding_set(
                                recording, None, position_decoding,
                                position_shift, property_name, func, min_bin_count,
                                speed_filter_kwargs=speed_filter_kwargs
                            )
                        )

            property_maps.append(animal_similarity_maps)

        return property_maps

    @staticmethod
    def compute_all_property_maps(all_recordings, position_decoding_name, min_bin_count, position_shift=0,
                                  position_decoding_var_param=None,
                                  subsampling_n_samples=None,
                                  estimators=('mean', 'median'),
                                  property_names=('peak_distance', 'log_peak_value', 'peak_value_normalised'),
                                  speed_filter_kwargs=None):

        property_map_names = []
        multiple_property_maps = []
        for func in estimators:
            for property_name in property_names:
                property_map_names.append(func + '_' + property_name)
                multiple_property_maps.append(
                    BayesianPositionDecodingPosteriorProperties.compute_property_maps(
                        all_recordings, position_decoding_name, min_bin_count, func, property_name,
                        position_shift=position_shift,
                        position_decoding_var_param=position_decoding_var_param,
                        subsampling_n_samples=subsampling_n_samples,
                        speed_filter_kwargs=speed_filter_kwargs
                    )
                )

        return multiple_property_maps, property_map_names


class DecodingStabilitySamplingRelationship(object):

    @staticmethod
    def get_decoding_accuracy_maps(all_recordings, position_decoding_name, min_bin_count, position_shift):
        """
        Returns a list of dicts, where list elements are results for each animal and
        dict elements are decoding accuracy arrays for the experiment_id specified by key.
        """
        return BayesianPositionDecodingPosteriorProperties.compute_all_property_maps(
                all_recordings, position_decoding_name, min_bin_count, position_shift=position_shift,
                estimators=('median',), property_names=('peak_distance',)
            )[0][0]

    @staticmethod
    def get_posterior_peak_maps(all_recordings, position_decoding_name, min_bin_count, position_shift):
        """
        Returns a list of dicts, where list elements are results for each animal and
        dict elements are posterior peak height arrays for the experiment_id specified by key.
        """
        return BayesianPositionDecodingPosteriorProperties.compute_all_property_maps(
                all_recordings, position_decoding_name, min_bin_count, position_shift=position_shift,
                estimators=('median',), property_names=('log_peak_value',)
            )[0][0]

    @staticmethod
    def get_ratemap_stability_maps(all_recordings, min_bin_samples):

        stability_maps = []
        for recordings in all_recordings:

            animal_stability_maps = {}
            for recording in recordings[:4]:

                map_1 = []
                map_2 = []
                for unit in recording.units:
                    if unit['analysis']['category'] == 'place_cell':
                        continue

                    map_1.append(unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['odd'])
                    map_2.append(unit['analysis']['spatial_ratemaps']['spike_rates_minutes']['even'])

                animal_stability_maps[recording.info['experiment_id']] = \
                    spatial_correlation(np.stack(map_1, axis=2), np.stack(map_2, axis=2),
                                        min_included_value=0.01, bin_wise=True,
                                        min_bin_samples=min_bin_samples)

            stability_maps.append(animal_stability_maps)

        return stability_maps

    @staticmethod
    def get_sampling_maps(all_recordings):

        sampling_maps = []
        for recordings in all_recordings:

            animal_sampling_maps = {}
            for recording in recordings[:4]:

                spatial_ratemap = SpatialRatemap(
                    recording.position['xy'], np.array([0.1]), recording.position['sampling_rate'],
                    spatial_window=(0, recording.info['arena_size'][0], 0, recording.info['arena_size'][1]),
                    xy_mask=recording.position['analysis']['ratemap_speed_mask'],
                    bin_size=Params.spatial_ratemap['bin_size']
                )

                animal_sampling_maps[recording.info['experiment_id']] = spatial_ratemap.dwell_time

            sampling_maps.append(animal_sampling_maps)

        return sampling_maps


class PlaceFieldPeakDistribution(object):

    experiment_ids = ('exp_scales_a', 'exp_scales_b', 'exp_scales_c', 'exp_scales_d')

    @staticmethod
    def binned_peak_counts(field_x, field_y, spatial_window, map_shape, bin_size):
        return np.histogram2d(
            field_x, field_y,
            SpatialRatemap.compute_bin_edges_from_spatial_window_and_shape(map_shape, spatial_window, bin_size)
        )[0].T

    @staticmethod
    def compute_field_density_around_each_position_bin(field_x_coords, field_y_coords, experiment_id, map_shape,
                                                       map_bin_size, kernel_size_in_bins, kernel_size_in_values,
                                                       fields_in_environment, fields_across_environments,
                                                       idx_map_bins_included):

        peak_map = PlaceFieldPeakDistribution.binned_peak_counts(
            field_x_coords, field_y_coords, spatial_windows[experiment_id],
            map_shape, map_bin_size
        ).astype(np.float32)

        peak_map_percentage_of_recording = peak_map / fields_in_environment
        peak_map_percentage_of_recording = convolve(peak_map_percentage_of_recording,
                                                    np.ones((kernel_size_in_bins, kernel_size_in_bins)),
                                                    normalize_kernel=False, fill_value=np.nan)
        peak_map_percentage_of_recording = \
            peak_map_percentage_of_recording / (kernel_size_in_values ** 2) * (100 ** 2)

        peak_map_percentage_of_recordings = peak_map / fields_across_environments
        peak_map_percentage_of_recordings = convolve(peak_map_percentage_of_recordings,
                                                     np.ones((kernel_size_in_bins, kernel_size_in_bins)),
                                                     normalize_kernel=False, fill_value=np.nan)
        peak_map_percentage_of_recordings = \
            peak_map_percentage_of_recordings / (kernel_size_in_values ** 2) * (100 ** 2)

        x_centers, y_centers = \
            SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(map_shape,
                                                                             spatial_windows[experiment_id])

        ix_included = []
        iy_incldued = []
        environments = []
        distances = []
        field_peaks_of_environment = []
        field_peaks_of_total = []
        x_coords = []
        y_coords = []
        experiment_ids = []

        for ix in range(x_centers.size):
            for iy in range(y_centers.size):

                # Do not include values too close to edges as bins are incomplete
                if not idx_map_bins_included[iy, ix] or np.isnan(peak_map_percentage_of_recording[iy, ix]):
                    continue

                ix_included.append(ix)
                iy_incldued.append(iy)

                distance = snippets.compute_distance_to_nearest_wall(
                    (x_centers[ix], y_centers[iy]),
                    (spatial_windows[experiment_id][1], spatial_windows[experiment_id][3])
                )
                environments.append(experiment_id_substitutes[experiment_id])
                distances.append(distance)
                field_peaks_of_environment.append(peak_map_percentage_of_recording[iy, ix])
                field_peaks_of_total.append(peak_map_percentage_of_recordings[iy, ix])
                x_coords.append(x_centers[ix])
                y_coords.append(y_centers[iy])
                experiment_ids.append(experiment_id)

        return pd.DataFrame({
            'ix': ix_included,
            'iy': iy_incldued,
            'environment': environments,
            'distance to wall (cm)': np.array(distances),
            'field peak proportion of environment per m^2': np.array(field_peaks_of_environment),
            'field peak proportion of total per m^2': np.array(field_peaks_of_total),
            'x_coord': np.array(x_coords),
            'y_coord': np.array(y_coords),
            'experiment_id': experiment_ids
        })

    @staticmethod
    def compute(all_recordings, df_fields, df_units, method, bin_size=20, combined=False):

        if method == 'peaks':
            method_x, method_y = ('peak_x', 'peak_y')
        elif method == 'centroids':
            method_x, method_y = ('centroid_x', 'centroid_y')
        else:
            raise ValueError('Unknown method {}'.format(method))

        map_bin_size = Params.spatial_ratemap['bin_size']

        if bin_size % map_bin_size != 0 or int(bin_size / map_bin_size) % 2 == 0:
            raise ValueError('Input bin_size must divide by map bin size to odd integer.')

        kernel_size = int(bin_size / map_bin_size)

        # Collect sampling maps
        sampling_maps = DecodingStabilitySamplingRelationship.get_sampling_maps(all_recordings)

        # Only keep fields belonging to place cells
        df_fields = df_fields[df_fields['unit'].isin(np.where(df_units['category'] == 'place_cell')[0])]

        # Only keep fields not in exp_scales_a2
        df_fields = df_fields[df_fields['experiment_id'] != 'exp_scales_a2']

        # Create a copy of df_fields with only the relevant columns
        df_fields = df_fields[['animal', 'experiment_id'] + [method_x, method_y]].copy(deep=True)

        if combined:

            # Compute data combining fields across animals
            dfs = []
            for experiment_id in PlaceFieldPeakDistribution.experiment_ids:

                idx = df_fields['experiment_id'] == experiment_id

                # Compute dwell time per 1 m2 bin as average of bin_size x bin_size cm bins
                sampling_map = [sm[experiment_id] for sm in sampling_maps]
                sampling_map = np.sum(np.stack(sampling_map, axis=2), axis=2)
                sampling_map = convolve(sampling_map, np.ones((kernel_size, kernel_size)), normalize_kernel=False,
                                        fill_value=np.nan)
                sampling_map = sampling_map / ((kernel_size * map_bin_size) ** 2) * (100 ** 2)

                # Compute proportion of peaks per 1 m2 bin as average of bin_size x bin_size cm bins

                df = PlaceFieldPeakDistribution.compute_field_density_around_each_position_bin(
                    df_fields[idx][method_x], df_fields[idx][method_y], experiment_id, sampling_map.shape,
                    map_bin_size, kernel_size, bin_size, np.sum(idx), df_fields.shape[0],
                    ~np.isnan(sampling_map)
                )
                df['dwell time per m^2 (s)'] = sampling_map[df['iy'].values, df['ix'].values]

                dfs.append(df)

        else:

            # Compute data separately for each animal
            dfs = []
            for i_animal, recordings in enumerate(all_recordings):

                fields_in_recordings = np.sum(df_fields['animal'] == recordings[0].info['animal'])

                for experiment_id in PlaceFieldPeakDistribution.experiment_ids:

                    animal = recordings[0].info['animal']

                    # Compute dwell time per 1 m2 bin as average of bin_size x bin_size cm bins
                    sampling_map = sampling_maps[i_animal][experiment_id]
                    sampling_map = convolve(sampling_map, np.ones((kernel_size, kernel_size)), normalize_kernel=False,
                                            fill_value=np.nan)
                    sampling_map = sampling_map / ((kernel_size * map_bin_size) ** 2) * (100 ** 2)

                    # Compute proportion of peaks per 1 m2 bin as average of bin_size x bin_size cm bins

                    idx = (df_fields['animal'] == animal) & (df_fields['experiment_id'] == experiment_id)
                    fields_in_recording = np.sum(idx)

                    df = PlaceFieldPeakDistribution.compute_field_density_around_each_position_bin(
                        df_fields[idx][method_x], df_fields[idx][method_y], experiment_id, sampling_map.shape,
                        map_bin_size, kernel_size, bin_size, fields_in_recording, fields_in_recordings,
                        ~np.isnan(sampling_map)
                    )
                    df['dwell time per m^2 (s)'] = sampling_map[df['iy'].values, df['ix'].values]
                    df['animal'] = animal

                    dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True, sort=True)


class PopulationVectorChangeRate(object):
    """
    This version uses spatial smoothing of spike rates.
    """
    # change_measure_methods = ('difference', 'difference_normalised', 'euclidean', 'cosine')
    change_measure_methods = ('euclidean',)

    @staticmethod
    def assign_axes_values_to_wall_alignment_values(values_along_x_axis, values_along_y_axis, x_contributions, y_contributions, experiment_id):
        """

        :param values_along_x_axis:
        :param values_along_y_axis:
        :param experiment_id:
        :return: values_along_short_wall, values_along_long_wall
        """
        if experiment_id == 'exp_scales_a':
            return values_along_x_axis, values_along_y_axis, x_contributions, y_contributions
        elif experiment_id == 'exp_scales_b':
            return values_along_y_axis, values_along_x_axis, y_contributions, x_contributions
        elif experiment_id == 'exp_scales_c':
            return values_along_x_axis, values_along_y_axis, x_contributions, y_contributions
        elif experiment_id == 'exp_scales_d':
            return values_along_y_axis, values_along_x_axis, y_contributions, x_contributions
        else:
            raise ValueError('Unable to handle experiment_id {}'.format(experiment_id))

    @staticmethod
    def compute_population_vector_change_rate(
            population_vectors: np.ndarray,
            xy: np.ndarray,
            idx_to_use: np.ndarray,
            timestamps: np.ndarray,
            euclidean_step_distance: float = 1,
            zscore: bool = False,
            spatial_smoothing: bool = False,
            change_measure_method: str = 'difference'
    ):

        linearised_position = np.nancumsum(
            np.concatenate(([0], snippets.euclidean_for_array(xy[:-1, :], xy[1:, :])))
        )

        target_linearised_positions = np.arange(0, np.nanmax(linearised_position), euclidean_step_distance)

        if spatial_smoothing:

            # Interpolate population vectors to desired sampling with intermediate smoothing at very high sampling

            intermediate_linearised_position = np.arange(0, np.nanmax(linearised_position), 0.1)

            interpolated_population_vectors = []
            for i_unit in range(population_vectors.shape[1]):

                unit_rates = population_vectors[:, i_unit]
                intermediate_interpolated_unit_rate = \
                    interp1d(linearised_position, unit_rates)(intermediate_linearised_position)

                intermediate_interpolated_unit_rate = np.convolve(
                    intermediate_interpolated_unit_rate, Gaussian1DKernel(31).array, mode='same'
                )

                interpolated_unit_rate = \
                    interp1d(intermediate_linearised_position,
                             intermediate_interpolated_unit_rate)(target_linearised_positions)
                interpolated_population_vectors.append(interpolated_unit_rate)

            interpolated_population_vectors = np.stack(interpolated_population_vectors, axis=1)

        else:

            # Interpolate population vectors to desired sampling
            interpolated_population_vectors = \
                interp1d(linearised_position, population_vectors, axis=0)(target_linearised_positions)

        interpolated_xy = interp1d(linearised_position, xy, axis=0)(target_linearised_positions)

        interpolated_timestamps = interp1d(linearised_position, timestamps)(target_linearised_positions)

        # ZScore the population vectors for each unit if requested
        if zscore:
            unit_means = np.nanmean(interpolated_population_vectors, axis=0)[None, :]
            unit_std = np.nanstd(interpolated_population_vectors, axis=0)[None, :]

            # Avoid division by 0 for elements where all values are 0
            idx = (unit_std == 0) & (unit_means == 0)
            if np.any(idx):
                unit_std[idx] = 1

            interpolated_population_vectors = (interpolated_population_vectors - unit_means) / unit_std

        if change_measure_method == 'difference':
            change_rate = \
                np.mean(np.abs(np.diff(interpolated_population_vectors, axis=0)), axis=1) / euclidean_step_distance
        elif change_measure_method == 'difference_normalised':
            mean_rate = (interpolated_population_vectors[:-1, :] + interpolated_population_vectors[1:, :]) / 2
            mean_rate[mean_rate < 1] = np.nan
            change_rate = (
                    np.nanmean(np.abs(np.diff(interpolated_population_vectors, axis=0)) / mean_rate, axis=1)
                    / euclidean_step_distance
            )
        elif change_measure_method == 'euclidean':
            change_rate = \
                snippets.euclidean_for_array(interpolated_population_vectors[:-1, :],
                                             interpolated_population_vectors[1:, :]) / euclidean_step_distance
        elif change_measure_method == 'cosine':
            change_rate = (
                np.sum(interpolated_population_vectors[:-1, :] * interpolated_population_vectors[1:, :], axis=1)
                / (np.linalg.norm(interpolated_population_vectors[:-1, :], axis=1)
                   * np.linalg.norm(interpolated_population_vectors[1:, :], axis=1))
            ) / euclidean_step_distance
        else:
            raise ValueError('Unknown change_measure_method {}'.format(change_measure_method))

        # Compute the absolute change along x and y axis
        xy_change = np.abs(interpolated_xy[1:, :] - interpolated_xy[:-1, :])

        # Compute contributions of movement in either axis to change rate
        # xy_contributions = xy_change / np.sqrt(np.sum(xy_change ** 2, axis=1, keepdims=True))
        xy_contributions = xy_change
        x_change_rate = change_rate * xy_contributions[:, 0]
        y_change_rate = change_rate * xy_contributions[:, 1]

        # Compute movement direction
        movement_direction = Recording.compute_movement_direction(interpolated_xy)

        # Find indices to use after interpolation
        interpolated_idx_to_use = \
            interp1d(linearised_position, idx_to_use.astype(np.float32))(target_linearised_positions)
        interpolated_idx_to_use = interpolated_idx_to_use == 1.0

        # Skip the first element of interpolated_xy and interpolated_idx_to_use and movement_direction
        # to match the sampling of the firing rate change measures
        interpolated_xy = interpolated_xy[1:]
        interpolated_idx_to_use = interpolated_idx_to_use[1:]
        movement_direction = movement_direction[1:]
        interpolated_timestamps = interpolated_timestamps[1:]

        # Select subset of data based on interpolated_idx_to_use
        interpolated_xy = interpolated_xy[interpolated_idx_to_use, :]
        change_rate = change_rate[interpolated_idx_to_use]
        x_change_rate = x_change_rate[interpolated_idx_to_use]
        y_change_rate = y_change_rate[interpolated_idx_to_use]
        xy_contributions = xy_contributions[interpolated_idx_to_use, :]
        movement_direction = movement_direction[interpolated_idx_to_use]
        interpolated_timestamps = interpolated_timestamps[interpolated_idx_to_use]

        return interpolated_xy, change_rate, x_change_rate, y_change_rate, movement_direction, \
               xy_contributions[:, 0], xy_contributions[:, 1], interpolated_timestamps

    @staticmethod
    def get_dataframe(all_recordings, min_speed: float = 10, max_speed: float = 10 ** 6,
                      smoothing: str = 'temporal', verbose: bool = True):
        """

        :param all_recordings:
        :param min_speed:
        :param max_speed:
        :param smoothing: ('temporal', 'spatial', None)
        :param verbose:
        :return:
        """

        change_rates = {m: [] for m in PopulationVectorChangeRate.change_measure_methods}
        change_rates_along_short_wall = {m: [] for m in PopulationVectorChangeRate.change_measure_methods}
        change_rates_along_long_wall = {m: [] for m in PopulationVectorChangeRate.change_measure_methods}
        contributions_rates_along_short_wall = []
        contributions_rates_along_long_wall = []
        x_coords = []
        y_coords = []
        distances = []
        animals = []
        experiment_ids = []
        environments = []
        movement_directions = []
        interpolated_timestamps_list = []
        for recordings in all_recordings:
            for i_recording, recording in enumerate(recordings[:4]):

                if verbose:
                    print('Computing firing rates for animal {} experiment {}'.format(
                        recording.info['animal'], recording.info['experiment_id']
                    ))

                idx_samples_in_environment = (
                    np.all(recording.position['xy'] > 0, axis=1)
                    & (recording.position['xy'][:, 0] <= recording.info['arena_size'][0])
                    & (recording.position['xy'][:, 1] <= recording.info['arena_size'][1])
                )
                idx_samples_with_correct_speed = ((recording.position['speed'] > min_speed)
                                                  & (recording.position['speed'] < max_speed))
                idx_position_samples_to_use = idx_samples_in_environment & idx_samples_with_correct_speed

                # Compute population vector
                population_vectors = []
                for i_unit, recordings_unit in enumerate(recordings.units):
                    if recordings.first_available_recording_unit(i_unit)['analysis']['category'] != 'place_cell':
                        continue

                    unit = recordings_unit[i_recording]

                    if unit is None:
                        timestamps = np.array([1])
                    else:
                        timestamps = unit['timestamps']

                    if smoothing == 'temporal':
                        spike_histogram = count_spikes_in_sample_bins(
                            timestamps, recording.position['sampling_rate'],
                            0, recording.position['xy'].shape[0] - 1,
                            sum_samples=5,
                            sum_samples_kernel='gaussian'
                        )
                    else:
                        spike_histogram = count_spikes_in_sample_bins(
                            timestamps, recording.position['sampling_rate'],
                            0, recording.position['xy'].shape[0] - 1
                        )

                    spike_histogram *= 0 if unit is None else recording.position['sampling_rate']

                    population_vectors.append(spike_histogram)

                population_vectors = np.stack(population_vectors, axis=1)
                timestamps = recording.position['timestamps']

                # Compute results
                sample_xy_previous = None
                movement_direction_previous = None
                interpolated_timestamps_previous = None
                x_contributions_previous = None
                y_contributions_previous = None
                for change_measure_method in PopulationVectorChangeRate.change_measure_methods:

                    xy = recording.position['xy']
                    xy = savgol_filter(xy, 31, 5, axis=0)

                    (sample_xy, change_rate, x_change_rate, y_change_rate,
                     movement_direction, x_contributions, y_contributions,
                     interpolated_timestamps) = \
                        PopulationVectorChangeRate.compute_population_vector_change_rate(
                            population_vectors, xy, idx_position_samples_to_use, timestamps,
                            change_measure_method=change_measure_method,
                            spatial_smoothing=(True if smoothing == 'spatial' else False)
                        )

                    if sample_xy_previous is not None:
                        assert np.allclose(sample_xy_previous, sample_xy, equal_nan=True)
                        assert np.allclose(movement_direction_previous, movement_direction, equal_nan=True)
                        assert np.allclose(interpolated_timestamps_previous, interpolated_timestamps, equal_nan=True)
                        assert np.allclose(x_contributions_previous, x_contributions, equal_nan=True)
                        assert np.allclose(y_contributions_previous, y_contributions, equal_nan=True)
                    else:
                        sample_xy_previous = sample_xy
                        movement_direction_previous = movement_direction
                        interpolated_timestamps_previous = interpolated_timestamps
                        x_contributions_previous = x_contributions
                        y_contributions_previous = y_contributions

                    (change_along_short_wall, change_along_long_wall,
                     contributions_along_short_wall, contributions_along_long_wall) = \
                        PopulationVectorChangeRate.assign_axes_values_to_wall_alignment_values(
                            x_change_rate, y_change_rate, x_contributions, y_contributions,
                            recording.info['experiment_id']
                        )

                    # Append to list across animals and recordings
                    change_rates[change_measure_method].append(change_rate)
                    change_rates_along_short_wall[change_measure_method].append(change_along_short_wall)
                    change_rates_along_long_wall[change_measure_method].append(change_along_long_wall)

                contributions_rates_along_short_wall.append(contributions_along_short_wall)
                contributions_rates_along_long_wall.append(contributions_along_long_wall)
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
                interpolated_timestamps_list.append(interpolated_timestamps)

        df = pd.DataFrame({
            'animal': np.concatenate(animals),
            'environment': np.concatenate(environments),
            'experiment_id': np.concatenate(experiment_ids),
            'x_coord': np.concatenate(x_coords),
            'y_coord': np.concatenate(y_coords),
            'distance to wall (cm)': np.concatenate(distances),
            'contributions along short wall': np.concatenate(contributions_rates_along_short_wall),
            'contributions along long wall': np.concatenate(contributions_rates_along_long_wall),
            'direction': np.concatenate(movement_directions),
            'timestamp': np.concatenate(interpolated_timestamps_list)
        })

        for change_method in PopulationVectorChangeRate.change_measure_methods:
            df['rate change\n({})'.format(change_method)] = \
                np.concatenate(change_rates[change_method])
            df['rate change along short wall\n({})'.format(change_method)] = \
                np.concatenate(change_rates_along_short_wall[change_method])
            df['rate change along long wall\n({})'.format(change_method)] = \
                np.concatenate(change_rates_along_long_wall[change_method])

        compute_distances_to_landmarks(df, np.stack((df['x_coord'].values, df['y_coord'].values), axis=1))

        return df
