
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
import warnings


def get_fitted_params_dict_from_fitted_model(fitted_model):
    """Returns a dictionary with all the parameters of the fitted model.

    :param fitted_model: an instance from classes in `astropy.modeling.models`
    :return: fitted_params
    :rtype: dict
    """
    return {key: value for key, value in zip(fitted_model.param_names, fitted_model.parameters)}


def fit_1d_gaussian(x, y, amplitude=1, mean=0, stddev=1, bounds=None, ignore_bad_fit_warnings=False, **kwargs):
    """Returns a fitted multiple Gaussian2D models and its extracted parameters.

    Note, if you get the message of "WARNING: The fit may be unsuccessful", try increasing
    number of allowed iterations with argument `maxiter=1000`.

    :param numpy.ndarray x: x values of data to be fitted
    :param numpy.ndarray y: y values of data to be fitted
    :param float amplitude: starting model amplitude (default is 1)
    :param float mean: starting model mean (default is 0)
    :param float stddev: starting model stddev (default is 1)
    :param dict bounds: A dictionary {parameter_name: value} of lower and upper bounds of parameters.
        Keys are parameter names. Values are a list or a tuple of length 2 giving the desired range for the parameter.
    :param bool ignore_bad_fit_warnings: if True, warnings about bad fit are suppressed (default is False)
    :param kwargs: any additional keyword arguments are passed on to call of
        :py:class:`astropy.modeling.fitting.LevMarLSQFitter` for fitting (e.g. maxiter=1000).
    :return: fitted_model, fitted_params
    """

    if bounds is None:
        bounds = {}
    gaussian_model = models.Gaussian1D(amplitude=amplitude, mean=mean, stddev=stddev, bounds=bounds)

    # Fit the gaussians to data
    fitter = fitting.LevMarLSQFitter()
    if ignore_bad_fit_warnings:
        # Filter unsuccessful fit warnings if requested
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='The fit may be unsuccessful')
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='overflow encountered in multiply')
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='divide by zero encountered in double_scalars')
            fitted_model = fitter(gaussian_model, x, y, **kwargs)
    else:
        fitted_model = fitter(gaussian_model, x, y, **kwargs)

    return fitted_model, get_fitted_params_dict_from_fitted_model(fitted_model)


def constrain_2d_gaussian_theta(fitted_model):
    """Corrects the angle parameter of a fitted model to be in range -pi to pi.

    :param fitted_model: an instance from classes in `astropy.modeling.models` or
        an element form :py:func:`fit_2d_gaussians_to_2d_data` output
    """

    theta = fitted_model.theta.value % (2 * np.pi)
    if theta > np.pi:
        theta = theta - np.pi

    fitted_model.theta.value = theta


def get_2d_gaussian_major_axis_theta_from_x_axis_from_parameters(fitted_params, symmetric=True):
    """Returns the angle in radians between x-axis and major axis of the 2DGaussian model.

    Angle is increasing clockwise from x-axis and 0 when pointing west.

    Requires a dictionary of parameters of :py:class:`astropy.modeling.models.Gaussian2D` instance.

    :param dict fitted_params: an element form :py:func:`fit_2d_gaussians_to_2d_data` second output
    :param bool symmetric: if True (default), angle is constrained to between 0 and pi, as gaussian
        is symmetric over both axes.
    :return: theta
    :rtype: float
    """
    theta = fitted_params['theta'] % (2 * np.pi)
    if theta > np.pi:
        theta = theta - np.pi
    if fitted_params['x_stddev'] < fitted_params['y_stddev']:
        theta = theta - np.pi / 2

    if symmetric:
        theta = (theta - np.pi) % np.pi

    return float(theta)


def get_2d_gaussian_major_axis_theta_from_x_axis(fitted_model):
    """Returns the angle in radians between x-axis and major axis of the 2DGaussian model.

    Angle is increasing clockwise from x-axis and 0 when pointing west.

    Requires :py:class:`astropy.modeling.models.Gaussian2D` instance as input.

    :param fitted_model: an element form :py:func:`fit_2d_gaussians_to_2d_data` first output
    :return: theta
    :rtype: float
    """
    return get_2d_gaussian_major_axis_theta_from_x_axis_from_parameters(
        get_fitted_params_dict_from_fitted_model(fitted_model)
    )


def fit_2d_gaussians_to_2d_data(data, init_params, data_x_values=None, data_y_values=None,
                                mask_nans=True, constrain_theta=True, ignore_bad_fit_warnings=False,
                                **kwargs):
    """Returns a fitted multiple Gaussian2D models and its extracted parameters.

    This function uses internally :py:mod:`astropy` Compound Model method to combine as many
    gaussians as specified by init_params. These fitted models are split in the output,
    but can be combined again with the same method as described for :py:mod:`astropy`.

    Note, if you get the message of "WARNING: The fit may be unsuccessful", try increasing
    number of allowed iterations with argument `maxiter=1000`.

    :param numpy.ndarray data: shape (n_ybins, n_xbins) array
    :param list init_params: list of dictionaries with initialization parameters.
        The number of elements in init_params specifies how many gaussians to fit.
        Each element is passed on to :py:class:`astropy.modeling.models.Gaussian2D` as **kwargs.
        See :py:class:`astropy.modeling.models.Gaussian2D` documentation for optional arguments.
    :param numpy.array data_x_values: values corresponding to data positions along second dimension (x)
    :param numpy.array data_y_values: values corresponding to data positions along first dimension (y)
    :param bool mask_nans: if True (default) a copy of data is created for fitting as a
        :py:class:`numpy.ma.array` with mask equal to `numpy.isnan(data)`.
    :param bool constrain_theta: if True (default), output model theta is corrected
        to be in range of -pi to pi. Note, the origin of the angle can vary between models.
        Use :py:func:`get_2d_gaussian_major_axis_theta_from_x_axis_from_parameters` for well defined angle.
    :param bool ignore_bad_fit_warnings: stops bad fit related warnigns being printed.
    :param kwargs: any additional keyword arguments are passed on to call of
        :py:class:`astropy.modeling.fitting.LevMarLSQFitter` for fitting (e.g. maxiter=1000).
    :return:fitted_models, fitted_params
    """

    # Mask numpy.nan values in data if requested.
    if mask_nans:
        data = np.ma.array(data, mask=np.isnan(data))

    # Construct compound model of multiple gaussians if many initialisation parameters specified.
    sum_of_gaussians = models.Gaussian2D(**init_params[0])

    for i in range(1, len(init_params)):

        sum_of_gaussians += models.Gaussian2D(**init_params[i])

    # Construct x and y position values for fitting
    if data_x_values is None:
        data_x_values = np.arange(data.shape[1])

    if data_y_values is None:
        data_y_values = np.arange(data.shape[0])

    x, y = np.meshgrid(data_x_values, data_y_values)

    # Fit the gaussians to data
    fitter = fitting.LevMarLSQFitter()
    if ignore_bad_fit_warnings:
        # Filter unsuccessful fit warnings if requested
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='The fit may be unsuccessful')
            warnings.filterwarnings(action='ignore', category=RuntimeWarning,
                                    message='overflow encountered in multiply')
            fitted_models = fitter(sum_of_gaussians, x, y, data, **kwargs)
    else:
        fitted_models = fitter(sum_of_gaussians, x, y, data, **kwargs)

    # If only one gaussian, set it as a single element in a list for compatibility
    # with multiple gaussian use-case.
    if len(init_params) == 1:
        fitted_models = [fitted_models]

    # Loop across all fitted gaussian models
    fitted_params = []
    for fitted_model in fitted_models:

        # Constrain theta to between -pi and pi if requested
        if constrain_theta:
            constrain_2d_gaussian_theta(fitted_model)

        # Extract fitted parameters into dictionaries
        fitted_params.append(
            get_fitted_params_dict_from_fitted_model(fitted_model)
        )

    return fitted_models, fitted_params


def draw_model_contours_on_image(fitted_model, ax, levels, **kwargs):
    """Plots contour lines of a fitted model to a image axes provided.

    Infers the point locations on the image from the first image found as part
    of the axes and using its dimensions and extend parameters.

    :param fitted_model: an instance from classes in :py:mod:`astropy.modeling.models`
    :param ax: :py:class:`matplotlib.axes.Axes` instance with associate image (`ax.imshow`)
    :param int levels: number of lines or their positions, see :py:func:`matplotlib.pyplot.contour`
    :param kwargs: all these are passed on to :py:func:`matplotlib.pyplot.contour`
    """

    # Get image shape and axes values range
    image_shape = ax.get_images()[0].properties()['array'].shape
    left, right, bottom, top = ax.get_images()[0].properties()['extent']

    # Create x and y position values
    y_step = (bottom - top) / float(image_shape[0])
    x_step = (right - left) / float(image_shape[1])
    y, x = np.mgrid[top:bottom:y_step, left:right:x_step]

    points_from_model = fitted_model(x, y)
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning,
                                message='No contour levels were found within the data range')
        ax.contour(x, y, points_from_model, levels, **kwargs)


def fit_gaussians_to_fields(ratemap, field_ratemaps, mask_nans=True, bound_mean_to_field=True,
                            stddev_bounds_arena_ratio=(0.05, 0.33), **kwargs):
    """Returns gaussian models fit to ratemap for each field.

    The gaussian amplitudes are bound to between `0` and `2 * numpy.max(field_ratemap)`.

    The below example show the process of constructing ratemaps, extracting fields
    and then fitting gaussians to these fields. To use this code, one would need
    to specify a file and `unit_index` that exhibits place fields.

    :Example:

    >>> from barrylab_ephys_analysis.recording_io import Recording
    >>> from barrylab_ephys_analysis.spatial.ratemaps import SpatialRatemap
    >>> from barrylab_ephys_analysis.spatial.fields import detect_fields
    >>> from barrylab_ephys_analysis.models.gaussian import fit_gaussians_to_fields
    >>> from barrylab_ephys_analysis.models.gaussian import draw_model_contours_on_image
    >>> import matplotlib.pyplot as plt
    >>>
    >>> unit_index = 10
    >>>
    >>> recording = Recording('/data1/filename.nwb')
    >>> stability_ratemaps = SpatialRatemap.instantiate_for_first_last_half(
    >>>     recording.position['xy'], recording.units[unit_index]['timestamps'],
    >>>     recording.position_data_sampling_rate, 4)
    >>> stability_ratemap_smoothed_1 = stability_ratemaps[0].spike_rates_smoothed(n_bins=2)
    >>> stability_ratemap_smoothed_2 = stability_ratemaps[1].spike_rates_smoothed(n_bins=2)
    >>> ratemap = SpatialRatemap(recording.position['xy'], recording.units[10]['timestamps'],
    >>>                          recording.position_data_sampling_rate, 4)
    >>> smoothed_ratemap = ratemap.spike_rates_smoothed(n_bins=2)
    >>> field_ratemaps = detect_fields(
    >>>     smoothed_ratemap, (stability_ratemap_smoothed_1, stability_ratemap_smoothed_2),
    >>>     1, 0.1, {'min_area_bins': 10, 'min_peak_value': 2},
    >>>     {'min_stability': 0.25, 'max_relative_bins': 0.5,
    >>>      'stability_kwargs': {'min_included_value': 0.01, 'min_bins': 6}}
    >>> )
    >>> data_x_values, data_y_values = SpatialRatemap.compute_bin_centers_from_spatial_window_and_shape(
    >>>     smoothed_ratemap.shape, ratemap.spatial_window
    >>> )
    >>> fitted_models, fitted_params = fit_gaussians_to_fields(
    >>>     smoothed_ratemap, field_ratemaps, mask_nans=True,
    >>>     data_x_values=data_x_values, data_y_values=data_y_values, maxiter=1000
    >>> )
    >>>
    >>> fig, ax = plt.subplots(1,1)
    >>> SpatialRatemap.plot(smoothed_ratemap, ratemap.spatial_window, ax=ax)
    >>> for fitted_model in fitted_models:
    >>>     draw_model_contours_on_image(fitted_model, ax, 3)
    >>> plt.show()

    :param numpy.ndarray ratemap: shape (n_ybins, n_xbins) complete ratemap of a recording.
        The 2D-gaussians are combined to fit this array.
    :param list field_ratemaps: iterable with elements numpy.ndarray shape (n_ybins, n_xbins),
        with numpy.nan values for points outside the field.
        Each field_ratemap is used to get the initialisation parameters for a 2D-gaussian,
        specifically the coordinates and maximum value. The number of elements in field_ratemaps
        list determines the number of gaussians fit to ratemap.
    :param bool mask_nans: if True (default) a copy of data is created for fitting as a
        :py:class:`numpy.ma.array` with mask equal to `numpy.isnan(data)`.
    :param bool bound_mean_to_field: if True (default), 2DGaussian x_mean and y_mean are bound
        to values within the smallest bounding rectangle of the field in ratemap.
    :param tuple stddev_bounds_arena_ratio: if provided maximum y_stddev and x_stddev bounds
        are set to (stddev_bounds_arena_ratio[0] * (shortest side of the ratemap),
                    stddev_bounds_arena_ratio[1] * (shortest side of the ratemap)).
        If provided, data_x_values and data_y_values are used to compute the shortest side.
    :param kwargs: additional arguments are passed on to :py:func:`fit_2d_gaussians_to_2d_data`
    :return: fitted_models, fitted_params - as output by :py:func:`fit_2d_gaussians_to_2d_data`
    """

    # Mask numpy.nan values in data if requested.
    if mask_nans:
        ratemap = np.ma.array(ratemap, mask=np.isnan(ratemap))

    init_params = []
    for field_ratemap in field_ratemaps:

        field_ratemap = np.ma.array(field_ratemap, mask=np.isnan(field_ratemap))

        bounds = {'amplitude': (0, 2 * np.max(field_ratemap))}

        if bound_mean_to_field:

            x_min = min(np.where(np.sum(~np.isnan(field_ratemap), axis=0) > 0)[0])
            x_max = max(np.where(np.sum(~np.isnan(field_ratemap), axis=0) > 0)[0])
            y_min = min(np.where(np.sum(~np.isnan(field_ratemap), axis=1) > 0)[0])
            y_max = max(np.where(np.sum(~np.isnan(field_ratemap), axis=1) > 0)[0])

            bounds['x_mean'] = (
                kwargs['data_x_values'][x_min] if 'data_x_values' in kwargs else x_min,
                kwargs['data_x_values'][x_max] if 'data_x_values' in kwargs else x_max
            )
            bounds['y_mean'] = (
                kwargs['data_y_values'][y_min] if 'data_y_values' in kwargs else y_min,
                kwargs['data_y_values'][y_max] if 'data_y_values' in kwargs else y_max
            )

        if not (stddev_bounds_arena_ratio is None):

            if 'data_y_values' in kwargs:
                arena_min_edge = min(kwargs['data_x_values'][-1] - kwargs['data_x_values'][0],
                                     kwargs['data_y_values'][-1] - kwargs['data_y_values'][0])
            else:
                arena_min_edge = min(field_ratemap.shape)

            bounds['x_stddev'] = (stddev_bounds_arena_ratio[0] * arena_min_edge,
                                  stddev_bounds_arena_ratio[1] * arena_min_edge)
            bounds['y_stddev'] = (stddev_bounds_arena_ratio[0] * arena_min_edge,
                                  stddev_bounds_arena_ratio[1] * arena_min_edge)

        a, b = np.unravel_index(np.argmax(field_ratemap), field_ratemap.shape)
        init_params.append({
            'amplitude': np.max(field_ratemap),
            'x_mean': kwargs['data_x_values'][b] if 'data_x_values' in kwargs else b,
            'y_mean': kwargs['data_y_values'][a] if 'data_y_values' in kwargs else a,
            'bounds': bounds
        })

    return fit_2d_gaussians_to_2d_data(ratemap, init_params, **kwargs)
