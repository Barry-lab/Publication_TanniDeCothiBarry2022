
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm


def spatial_correlation(map_1, map_2, min_included_value=0.01, mask=None, min_bins=4, bin_wise=False,
                        min_bin_samples=0):
    """Returns Pearson correlation of values in map_1 and map_2.

    Bins are ignored where any value in either map falls below `min_included_value` parameter
    or has the value `numpy.nan`

    :param numpy.ndarray map_1: shape (n_xbins, n_ybins) or (n_xbins, n_ybins, n_maps)
    :param numpy.ndarray map_2: shape (n_xbins, n_ybins) or (n_xbins, n_ybins, n_maps)
    :param float min_included_value: minimum values for bin to be included. If value is below
        this in any map, that bin is excluded computation. Default is 0.01
    :param numpy.ndarray mask: shape (n_xbins, n_ybins) boolean array specifying which bins to use
    :param int min_bins: minimum  number of bins that must remain to correlate, otherwise
        returns (numpy.nan, numpy.nan). Default is 4.
    :param bool bin_wise: if True, r-value is computed `n_maps` of `map_1` and `map_2`.
    :param int min_bin_samples: minimum number of samples per bin that must be valid for computing
        the r-value in case where `bin_wise=True`. Bins with fewer valid samples are set to `numpy.nan`.
    :return: output from :py:func:`scipy.stats.pearsonr` or numpy.ndarray of shape `map_1.shape[:-1]` with rho values
    """

    if map_1.ndim == 2:
        map_1 = map_1[:, :, np.newaxis]
    if map_2.ndim == 2:
        map_2 = map_2[:, :, np.newaxis]

    if map_1.shape[1] != map_2.shape[1]:
        raise ValueError('Number of maps does not match for map_1 {} and map_2 {}'.format(map_1.shape, map_2.shape))
    if not (mask is None) and (mask.ndim != 2):
        raise ValueError('Incorrect mask shape {}'.format(mask.shape))
    if not (mask is None) and (mask.shape[0] != map_1.shape[0] or mask.shape[1] != map_1.shape[1]):
        raise ValueError('Mask shape {} does not match with map shape {}'.format(mask.shape, map_1.shape))

    # Put mask to correct shape so broadcasting works also with stacked ratemaps
    if not (mask is None):
        mask = mask[:, :, np.newaxis]

    # Create masks based on np.nan values
    ignore_1 = np.isnan(map_1)
    ignore_2 = np.isnan(map_2)

    # Combine np.nan masks with min_included_value mask
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered in less')
        ignore_1 = np.logical_or(ignore_1, map_1 < min_included_value)
        ignore_2 = np.logical_or(ignore_2, map_2 < min_included_value)

    # Combine ignore masks of map_1 and map_2 to union ignore mask
    ignore = np.logical_or(ignore_1, ignore_2)

    # Convert ignore mask to inclusion mask and combine with mask parameter if provided
    mask = ~ignore if mask is None else np.logical_and(~ignore, mask)

    # If bin_wise is not True, return pearsonr across the whole array
    if not bin_wise:

        # If no bins are included, abort and return (np.nan, np.nan)
        if np.count_nonzero(mask) < min_bins:
            return np.nan, np.nan

        return pearsonr(map_1[mask], map_2[mask])

    # Otherwise, compute pearson correlation r-value for between the two maps along the last dimension

    r_values = np.zeros(map_1.shape[:-1], dtype=np.float32) * np.nan

    for i_x in range(r_values.shape[1]):
        for i_y in range(r_values.shape[0]):

            if np.sum(mask[i_y, i_x, :]) < min_bin_samples:
                continue

            r_values[i_y, i_x] = pearsonr(map_1[i_y, i_x, np.where(mask[i_y, i_x, :])[0]],
                                          map_2[i_y, i_x, np.where(mask[i_y, i_x, :])[0]])[0]

    return r_values
