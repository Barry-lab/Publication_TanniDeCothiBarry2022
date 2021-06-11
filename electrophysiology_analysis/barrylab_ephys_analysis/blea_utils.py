
import numpy as np
from scipy.signal import convolve
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt


def check_if_dictionaries_match(d1, d2):
    """Recursively compares all fields of the dictionary

    str items are compared directly, all other items are cast to numpy.array for comparison.
    Therefore any items must be string or possible to cast into numpy array.

    :param dict d1:
    :param dict d2:
    :return: True if dictionaries match
    :rtype: bool
    """
    if len(d1) != len(d2):
        return False
    for key in d1:
        if not (key in d2):
            return False
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            return check_if_dictionaries_match(d1[key], d2[key])
        else:
            return np.allclose(np.array(d1[key]), np.array(d2[key]))


def argparse_to_kwargs(parser):
    """Converts args in parser to dictionary that can be used as `**kwargs` in function call

    Any individual arguments are returned as a

    :param parser: :py:class:`argparse.ArgumentParser` instance
    :return: kwargs, dictionary of parameters that can be used as **kwargs
    :rtype: dict
    """
    args = parser.parse_args()

    kwargs = {}

    arg_info = {vars(x)['dest']: {'nargs': vars(x)['nargs']} for x in parser._actions}

    for arg_name in vars(args):

        arg_value = getattr(args, arg_name)

        if not (arg_value is None):

            kwargs[arg_name] = arg_value

            # If argument is supposed to have only one value, return only that list element
            if arg_info[arg_name]['nargs'] == 1:
                kwargs[arg_name] = kwargs[arg_name][0]

    return kwargs


def smooth_signal_with_gaussian(signal, sigma):
    """Returns signal smoothed with gaussian kernel of size 10 * sigma

    :param numpy.ndarray signal: shape (n_samples,)
    :param float sigma: stdev of gaussian kernel in number of samples
    :return:
    """

    kernel_width = int(round(sigma * 10))
    kernel_width = kernel_width - 1 if kernel_width % 2 == 0 else kernel_width
    kernel = gaussian(kernel_width, sigma)

    return convolve(signal, kernel / sum(kernel), 'same')


def all_list_element_combinations_as_pairs(l):
    """Returns a list of tuples that contains element pairs from `l` in all possible combinations
    without repeats irrespective of order.

    :param list l: list to parse into pairs
    :return: element_pairs, list of tuples for each pair
    :rtype: list
    """
    element_pairs = []
    for i in range(len(l)):
        element_pairs += [(l[i], l[j]) for j in range(i, len(l))]

    return element_pairs


def batch_generator(num_items, batch_size):
    """Returns an iterable for slicing an iterable of length num_items with
    slices of maximum batch_size length.

    :param num_items: total length of the iterable
    :param batch_size: maximum length of a slice
    :return: iterable with elements (first_ind, last_ind)
    :rtype: iterable
    """
    for first_ind in range(0, num_items, batch_size):
        yield (first_ind, min(first_ind + batch_size, num_items))


def euclidean_distance_between_rows_of_matrix(x, y):
    """Returns the euclidean distance between each row of matrix x and matrix y

    :param numpy.ndarray x: shape (n, m) array
    :param numpy.ndarray y: shape (n, m) array
    :return: distances shape (n,)
    :rtype: numpy.ndarray
    """
    return np.sqrt(np.sum(np.power(x - y, 2), axis=1))


def optimal_row_and_column_count_for_subplots(n):
    """Returns the optimal number of rows and columns for a given number of subplots

    :param int n: number of subplots required
    :return: n_cols, n_rows
    """
    n_cols = 1
    n_rows = 1
    increase_next = 'cols'
    while n_rows * n_cols < n:
        if increase_next == 'cols':
            n_cols += 1
            increase_next = 'rows'
        elif increase_next == 'rows':
            n_rows += 1
            increase_next = 'cols'

    return n_cols, n_rows


def matplotlib_figure_with_n_plots(n_plots, figsize_per_subplot=None, **kwargs):
    """Returns a matplotlib figure and axes with optimal number of rows and columns.

    Any additional keyword arguments are passed on to :py:func:`plt.subplots`

    :param int n_plots: number of subplots required
    :param tuple figsize_per_subplot: (width, height) of each individual subplot on the figure.
        This overrides any 'figsize' keyword argument.
    :return: figure, axes
    """
    n_cols, n_rows = optimal_row_and_column_count_for_subplots(n_plots)

    if not (figsize_per_subplot is None):
        kwargs['figsize'] = (figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows)

    return plt.subplots(nrows=n_rows, ncols=n_cols, **kwargs)


def gray_arrays_to_rgb_sequence_array(arrays, start_rgb, end_rgb, normalise_input=False, normalise_output=True):
    """Returns an RGB array that is mean of grayscale arrays mapped to linearly spaced RGB colors in a range.

    :param list arrays: list of numpy.ndarrays of shape (N, M)
    :param tuple start_rgb: (R, G, B) mapping of first array in `arrays`
    :param tuple end_rgb: (R, G, B) mapping of last array in `arrays`
    :param bool normalise_input: if True, input arrays are normalised concurrently to max value of 1. Default is False.
    :param bool normalise_output: if True (default), output is normalised to range between 0 and 1.
    :return: rgb_sequence_array shape (N, M, 3)
    :rtype: numpy.ndarray
    """

    if normalise_input:
        max_gray_value = max([np.max(array) for array in arrays])
        arrays = [array / max_gray_value for array in arrays]

    colors = np.array([np.linspace(start, end, len(arrays)) for start, end in zip(start_rgb, end_rgb)]).T

    color_arrays = [color[np.newaxis, np.newaxis, :] * array[:, :, np.newaxis] for color, array in zip(colors, arrays)]

    rgb_sequence_array = np.mean(np.stack(color_arrays, axis=3), axis=3)

    if normalise_output:
        rgb_sequence_array = rgb_sequence_array / np.nanmax(rgb_sequence_array)

    return rgb_sequence_array


def intersect2d_along_rows(ar1, ar2, assume_unique=False, return_indices=False):
    """Functions exactly as :py:func:`numpy.intersect1d`, but treats rows as single values.

    Columns are  merged into single elements using a new dtype.

    See :py:func:`numpy.intersect1d` for description of inputs and outputs.

    Inputs must have 2 dimensions and same dtype.
    """
    # Ensure that arrays are C-contiguous
    ar1 = np.ascontiguousarray(ar1)
    ar2 = np.ascontiguousarray(ar2)

    # Ensure arrays match requirements
    assert ar1.dtype == ar2.dtype, 'Inputs must be same dtype'
    assert len(ar1.shape) == len(ar2.shape) == 2, 'Inputs must have 2 dimensions'

    # Apply numpy.intersect1d across rows, merging columns with a new dtype
    dtype = {'names': ['f{}'.format(i) for i in range(ar1.shape[1])],
             'formats': ar1.shape[1] * [ar1.dtype]}
    return np.intersect1d(ar1.view(dtype), ar2.view(dtype),
                          assume_unique=assume_unique,
                          return_indices=return_indices)


def isin_along_rows(ar1, ar2, assume_unique=False, invert=False):
    """Functions exactly as :py:func:`numpy.isin`, but treats rows as single values.

    Columns are  merged into single elements using a new dtype.

    See :py:func:`numpy.isin` for description of inputs and outputs.

    Inputs must have 2 dimensions and same dtype.
    """
    # Ensure that arrays are C-contiguous
    ar1 = np.ascontiguousarray(ar1)
    ar2 = np.ascontiguousarray(ar2)

    # Ensure arrays match requirements
    assert ar1.dtype == ar2.dtype, 'Inputs must be same dtype'
    assert len(ar1.shape) == len(ar2.shape) == 2, 'Inputs must have 2 dimensions'

    # Apply numpy.isin across rows, merging columns with a new dtype
    dtype = {'names': ['f{}'.format(i) for i in range(ar1.shape[1])],
             'formats': ar1.shape[1] * [ar1.dtype]}
    return np.isin(ar1.view(dtype), ar2.view(dtype), assume_unique=assume_unique, invert=invert)


def count_digits(n):
    """Returns the number of digits in the integer.

    :param int n: integer
    :return: n_digits
    :rtype: int
    """
    if n > 0:
        return int(np.log10(n)) + 1
    elif n == 0:
        return 1
    else:
        return int(np.log10(-n)) + 1  # +2 if you count the '-'


def isin_along_rows_merge_with_sum(ar1, ar2, assume_unique=False, invert=False):
    """Functions exactly as :py:func:`numpy.isin`, but treats rows as single values.

    Works only with positive (or zero) integer arrays that have two columns.

    Columns are merged into single element by adding `1` to first column and multiplying
    the resulting values by `10 ** n_digits` (where `n_digits` is the number of digits
    in the maximum value in the second column) and then adding the second column.

    See :py:func:`numpy.isin` for description of inputs and outputs.
    """
    # Ensure that arrays are C-contiguous
    ar1 = np.ascontiguousarray(ar1)
    ar2 = np.ascontiguousarray(ar2)

    # Ensure arrays match requirements
    assert np.min(ar1) >= 0 and np.min(ar2) >= 0, 'All input elements must be positive or zero'
    assert (np.issubdtype(ar1.dtype, np.integer)
            and np.issubdtype(ar2.dtype, np.integer)), 'Inputs must both be integers dtype'
    assert len(ar1.shape) == len(ar2.shape) == 2, 'Inputs must have 2 dimensions'
    assert ar1.shape[1] == ar2.shape[1] == 2, 'Inputs must have 2 columns'

    first_column_multiplier = 10 ** count_digits(max(np.max(ar1[:, 1]), np.max(ar2[:, 1])))
    ar1 = (ar1[:, 0] + 1) * first_column_multiplier + ar1[:, 1]
    ar2 = (ar2[:, 0] + 1) * first_column_multiplier + ar2[:, 1]

    if np.max(ar1) == np.iinfo(ar1.dtype).max or np.max(ar2) == np.iinfo(ar2.dtype).max:
        raise ValueError('Input values reach limit of dtype after merging columns into elements.')

    return np.isin(ar1, ar2, assume_unique=assume_unique, invert=invert)
