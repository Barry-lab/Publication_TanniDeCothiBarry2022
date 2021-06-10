import numpy as np

def bin_data(var_to_bin_by, bin_size, limits, var_to_bin = []):
        """
        Creates N-d histogram of var_to_bin_by using bins of size bin_size. If var_to_bin is provided then it creates a weighted histogram using var_to_bin as the weights. Hence if var_to_bin is spikes and var_to_bin_by is x-y position then it creates a spatial histogram of spike counts.

        Parameters
        ----------
        var_to_bin_by: array-like with shape n_dim x n_samples (i.e. if x-y position then n_dim = 2)
        bin_size: scalar or tuple according to the dimensions of var_to_bin_by. Suppling a scalar for dim(var_to_bin_by) > 1 uses equal bin_size in all dimensions
        limits: n_dim list of tuples for lower and upper limits i.e. [(x_min, x_max), (y_min,y_max)]

        Optional:

        var_to_bin: 1d array providing weights for the counts in the histogram (e.g. spikes)


        Returns
        -------
        binned_data: array-like

        """
        n_dim = len(var_to_bin_by)

        if np.size(bin_size) < n_dim:
            bin_size = np.repeat(bin_size, n_dim)

        bins = []
        for i in range(n_dim):
            bins.append(np.arange(limits[i][0], limits[i][1] + bin_size[i], bin_size[i]))

        if  len(var_to_bin) == 0:
            hst = np.histogramdd(var_to_bin_by, bins = bins)
        else:
            hst = np.histogramdd(var_to_bin_by, bins = bins, weights = var_to_bin, normed = False)

        return hst[0]