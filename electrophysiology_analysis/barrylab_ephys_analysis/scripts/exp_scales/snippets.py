
import numpy as np
from scipy.spatial.distance import euclidean

from barrylab_ephys_analysis.external.circstats import difference as circular_difference
from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap
from barrylab_ephys_analysis.recording_io import Recordings


def rename_last_recording_a2(recordings):

    if recordings[-1].info['experiment_id'] == 'exp_scales_a':
        recordings[-1].edit_info()['experiment_id'] = 'exp_scales_a2'
    elif recordings[-1].info['experiment_id'] == 'exp_scales_a2':
        pass
    else:
        print('Final recording not exp_scales_a, skipping renaming {} }{}'.format(
            recordings[-1].info['animal'], recordings[-1].info['rec_datetime']
        ))


def get_index_where_most_spikes_in_unit_list(unit_list):
    """Returns the position in the list of units that has the most spikes.

    :param unit_list: list of unit dictionaries
    :return: index
    :rtype: int
    """
    return np.argmax(Recordings.count_spikes_in_unit_list(unit_list))


def distance_from_line_defined_by_two_points(point, line_point_1, line_point_2):
    """Returns euclidean distance of a point to a line defined by two points.

    :param numpy.ndarray point: array specifying the query point
    :param numpy.ndarray line_point_1: array specifying one of the points defining the line
    :param numpy.ndarray line_point_2: array specifying one of the points defining the line
    :return: distance
    :rtype: float
    """

    return np.abs(np.cross(line_point_2 - line_point_1, line_point_1 - point)
                  / np.linalg.norm(line_point_2 - line_point_1))


def rotate_angle_axis_origin(angle, rotation, max_angle=np.pi):
    """Returns the value of the angle with origin shifted by rotation value.

    All values are in radians.

    Assumes minimum angle value is 0.

    :param float angle: original angle
    :param float rotation: amount of origin rotation
    :param float max_angle: angle range maximum value
    :return: angle
    :rtype:
    """
    return (angle + rotation) % max_angle


def angular_deviation_from_line_angle(query_angle, line_angle):
    """Returns the minimal difference between the `query_angle` and the `line_angle` or
    the opposite of `line_angle`.

    :param numpy.ndarray query_angle: shape (N,)
    :param numpy.ndarray line_angle: shape (N,)
    :return: angluar_deviation shape (N,)
    :rtype: numpy.ndarray
    """
    if not isinstance(query_angle, np.ndarray):
        query_angle = np.array(query_angle)
    if not isinstance(line_angle, np.ndarray):
        line_angle = np.array(line_angle)
    if query_angle.ndim == 0:
        query_angle = np.array([query_angle])
    if line_angle.ndim == 0:
        line_angle = np.array([line_angle])

    angular_difference = circular_difference(line_angle, query_angle)
    angular_differences = np.stack([(angular_difference + 2 * np.pi) % (2 * np.pi) - np.pi, angular_difference], axis=1)
    angular_deviation = angular_differences[np.arange(angular_differences.shape[0]),
                                            np.argmin(np.abs(angular_differences), axis=1)]

    return angular_deviation


def compute_distance_to_nearest_corner(xy, arena_size):
    """Returns euclidean distance to nearest corner of the arena.

    Assumes that one corner of the rectangular environment is (0, 0) and
    others are defined by environment shape limits set by arena_size.

    :param xy: array like (x_position, y_position)
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :return: distance_to_corner
    :rtype: float
    """

    return min(euclidean(np.array(xy), np.array([0, 0])),
               euclidean(np.array(xy), np.array([0, arena_size[1]])),
               euclidean(np.array(xy), np.array([arena_size[0], arena_size[1]])),
               euclidean(np.array(xy), np.array([arena_size[0], 0])))


def euclidean_for_array(arr1, arr2):
    """Returns euclidean distance between points along 1st dimension in both arrays
    """
    return np.sqrt(np.sum((arr1 - arr2) ** 2, axis=1))


def compute_distance_to_nearest_corner_for_array(xy, arena_size):
    """Returns euclidean distance to nearest corner of the arena.

    Assumes that one corner of the rectangular environment is (0, 0) and
    others are defined by environment shape limits set by arena_size.

    :param numpy.ndarray xy: shape (n_samples, 2) x_position, y_position
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :return: distance_to_corner shape (n_samples,)
    :rtype: numpy.ndarray
    """
    return np.min(np.stack((euclidean_for_array(xy, np.array([0, 0])[np.newaxis, :]),
                            euclidean_for_array(xy, np.array([0, arena_size[1]])[np.newaxis, :]),
                            euclidean_for_array(xy, np.array([arena_size[0], arena_size[1]])[np.newaxis, :]),
                            euclidean_for_array(xy, np.array([arena_size[0], 0])[np.newaxis, :])),
                           axis=1),
                  axis=1)


def compute_distance_to_nearest_wall(xy, arena_size):
    """Returns euclidean distance to nearest point in any of the walls defined by arena_size.

    Assumes that one corner of the rectangular environment is (0, 0) and
    others are defined by environment shape limits set by arena_size.

    :param xy: array like (x_position, y_position)
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :return: distance_to_wall
    :rtype: float
    """
    walls_by_points = ((np.array([0, 0]), np.array([0, arena_size[1]])),
                       (np.array([0, arena_size[1]]), np.array([arena_size[0], arena_size[1]])),
                       (np.array([arena_size[0], arena_size[1]]), np.array([arena_size[0], 0])),
                       (np.array([arena_size[0], 0]), np.array([0, 0])))
    return min([distance_from_line_defined_by_two_points(np.array(xy), *wall_by_points)
                for wall_by_points in walls_by_points])


def axis_of_nearest_wall(xy, arena_size):
    """Returns whether the nearest wall is along x (1) or y (0) axis.

    Assumes that one corner of the rectangular environment is (0, 0) and
    others are defined by environment shape limits set by arena_size.

    :param xy: array like (x_position, y_position)
    :param arena_size: array like (max_x_axis_value, max_y_axis_value)
    :return: wall_axis (0, for vertical; 1, for horizontal axis)
    :rtype: int
    """
    walls_by_points = ((np.array([0, 0]), np.array([0, arena_size[1]])),
                       (np.array([0, arena_size[1]]), np.array([arena_size[0], arena_size[1]])),
                       (np.array([arena_size[0], arena_size[1]]), np.array([arena_size[0], 0])),
                       (np.array([arena_size[0], 0]), np.array([0, 0])))
    nearest_wall_ind = np.argmin([distance_from_line_defined_by_two_points(np.array(xy), *wall_by_points)
                                  for wall_by_points in walls_by_points])
    return (0, 1, 0, 1)[int(nearest_wall_ind)]


def get_first_animal_recordings(all_recordings, animal):
    """Returns the first animal Recordings instance from a list of Recordings instances

    :param list all_recordings: list of :py:class:`recording_io.Recordings` instances
    :param str animal: value to look for in Recordings[0].info['animal']
    :return: recording_io.Recordings
    """
    for recordings in all_recordings:
        if recordings[0].info['animal'] == animal:
            return recordings


class SpatialMapBinCenterDistanceToWall(object):

    def __init__(self):

        self.computed = {}

    @staticmethod
    def compute(map_shape, spatial_window):

        data_x_values, data_y_values = SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(
            map_shape,
            spatial_window
        )

        distances = np.zeros(map_shape, dtype=np.float32)

        for ix, x_val in enumerate(data_x_values):
            for iy, y_val in enumerate(data_y_values):

                distances[iy, ix] = compute_distance_to_nearest_wall((x_val, y_val),
                                                                     (spatial_window[1], spatial_window[3]))

        return distances

    def get(self, map_shape, spatial_window, n_xbin=None, n_ybin=None):

        if not (map_shape in self.computed):

            self.computed[map_shape] = {}

        if not (spatial_window in self.computed[map_shape]):

            self.computed[map_shape][spatial_window] = \
                SpatialMapBinCenterDistanceToWall.compute(map_shape, spatial_window)

        if n_xbin is None and n_ybin is None:

            return self.computed[map_shape][spatial_window]

        else:

            return self.computed[map_shape][spatial_window][n_ybin, n_xbin]
