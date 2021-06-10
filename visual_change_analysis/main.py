import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from bin_data import bin_data

# import pixel data   
right_z_pixel_change = np.load("right_z_pixel_change.npy")
left_z_pixel_change = np.load("left_z_pixel_change.npy")
front_z_pixel_change = np.load("front_z_pixel_change.npy")

# average pixel change across front, left & right fovs
pixel_change = np.vstack((left_z_pixel_change, front_z_pixel_change, right_z_pixel_change)).mean(axis=0)

# import rate change data   
dat = pd.read_pickle("smooth_df_population_vector_change_with_other_variables.p")

# Clean the data (sequential data points are 1cm apart along trajectory)
dat = dat[dat.environment == 'D']
df = dat.filter(['animal', 'x_coord', 'y_coord', 'direction', 'timestamp'], axis=1)
dat = dat[~df.isnull().any(axis=1)]
good_pixel_ids = np.array(np.diff(dat.x_coord)**2 + np.diff(dat.y_coord)**2 < 1.01, dtype=bool)
pixel_change = pixel_change[good_pixel_ids]
good_rate_ids = np.append(False, good_pixel_ids)
turning_rate = np.abs(np.diff(dat['direction'])) % 360
turning_rate = turning_rate[good_pixel_ids]
dat = dat[good_rate_ids]

# z-score data
dat['rate change\n(euclidean)'] = (dat['rate change\n(euclidean)'] - np.mean(dat['rate change\n(euclidean)']))/np.std(dat['rate change\n(euclidean)'])
pixel_change = (pixel_change - np.mean(pixel_change))/np.std(pixel_change)

# Plot Occupancy
occupancy = bin_data([dat.x_coord, dat.y_coord], bin_size = 4, limits = [(0, 350), (0, 250)])
plt.imshow(occupancy.T, origin='upper', cmap=plt.get_cmap('jet'))
plt.title('Occupancy')
plt.show()

# Plot pixel change across space
pixel_change_map = bin_data([dat.x_coord, dat.y_coord], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = pixel_change) / occupancy
plt.imshow(pixel_change_map.T, origin='upper', cmap=plt.get_cmap('jet'))
plt.axis('off')
plt.clim([-1.5,1.5])
plt.title('Pixel Change Map')
plt.show()

# Plot firing rate change across space
rate_change_map = bin_data([dat.x_coord, dat.y_coord], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = dat['rate change\n(euclidean)']) / occupancy
plt.imshow(rate_change_map.T, origin='upper', cmap=plt.get_cmap('jet'))
plt.axis('off')
plt.clim([-1.5,1.5])
plt.title('Rate Change Map')
plt.show()

corr, _ = pearsonr(pixel_change, dat['rate change\n(euclidean)'])
print('Rate Change vs Pixel Change Pearson r = %.3f' % corr)

# Filter bits of trajectory by head direction
north_ids = (np.degrees(dat.direction) % 360 >= 315) | (np.degrees(dat.direction) % 360 < 45)
north_occupancy = bin_data([dat.x_coord[north_ids], dat.y_coord[north_ids]], bin_size = 4, limits = [(0, 350), (0, 250)])
south_ids = (np.degrees(dat.direction) % 360 >= 135) & (np.degrees(dat.direction) % 360 < 225)
south_occupancy = bin_data([dat.x_coord[south_ids], dat.y_coord[south_ids]], bin_size = 4, limits = [(0, 350), (0, 250)])
east_ids = (np.degrees(dat.direction) % 360 >= 45) & (np.degrees(dat.direction) % 360 < 135)
east_occupancy = bin_data([dat.x_coord[east_ids], dat.y_coord[east_ids]], bin_size = 4, limits = [(0, 350), (0, 250)])
west_ids = (np.degrees(dat.direction) % 360 >= 225) & (np.degrees(dat.direction) % 360 < 315)
west_occupancy = bin_data([dat.x_coord[west_ids], dat.y_coord[west_ids]], bin_size = 4, limits = [(0, 350), (0, 250)])

cmap = plt.get_cmap('jet')
cmap.set_bad('w',1.)

# Calculate pixel and rate change maps by heading direction
north_pix_map = bin_data([dat.x_coord[north_ids], dat.y_coord[north_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = pixel_change[north_ids]) / north_occupancy
south_pix_map = bin_data([dat.x_coord[south_ids], dat.y_coord[south_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = pixel_change[south_ids]) / south_occupancy
east_pix_map = bin_data([dat.x_coord[east_ids], dat.y_coord[east_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = pixel_change[east_ids]) / east_occupancy
west_pix_map = bin_data([dat.x_coord[west_ids], dat.y_coord[west_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = pixel_change[west_ids]) / west_occupancy
north_rat_map = bin_data([dat.x_coord[north_ids], dat.y_coord[north_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = dat['rate change\n(euclidean)'][north_ids]) / north_occupancy
south_rat_map = bin_data([dat.x_coord[south_ids], dat.y_coord[south_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = dat['rate change\n(euclidean)'][south_ids]) / south_occupancy
east_rat_map = bin_data([dat.x_coord[east_ids], dat.y_coord[east_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = dat['rate change\n(euclidean)'][east_ids]) / east_occupancy
west_rat_map = bin_data([dat.x_coord[west_ids], dat.y_coord[west_ids]], bin_size = 4, limits = [(0, 350), (0, 250)], var_to_bin = dat['rate change\n(euclidean)'][west_ids]) / west_occupancy

c_lo = -1.5
c_hi = 1.5

# Plot change maps filtered by direction
plt.subplot(3,3,2)
plt.title('Unfolded Pixel Change Map')
plt.imshow(west_pix_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,4)
plt.imshow(south_pix_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(pixel_change_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,6)
plt.imshow(north_pix_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,8)
plt.imshow(east_pix_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.show()

plt.subplot(3,3,2)
plt.title('Unfolded Rate Change Map')
plt.imshow(west_rat_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,4)
plt.imshow(south_rat_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,5)
plt.imshow(rate_change_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,6)
plt.imshow(north_rat_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.subplot(3,3,8)
plt.imshow(east_rat_map.T, origin='upper', cmap=cmap)
plt.clim([c_lo,c_hi])
plt.axis('off')
plt.show()