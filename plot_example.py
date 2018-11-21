"""
iris NEMO examples
"""
from siri_omen import *

# extract time series with correct location and dataset metadata
te = TimeSeriesExtractor('NORDIC_1h_20180901_20181029_grid_T_201809*.nc')

# extract one time series
# location_name and dataset_id are required attributes
cube1 = te.extract('sea_water_practical_salinity',
                   lon=18.9377, lat=60.5332, z=-20.0,
                   location_name='station1',
                   dataset_id='run001')

# extract another time series
cube2 = te.extract('sea_water_practical_salinity',
                   lon=19.5797, lat=61.0833, z=-30.0,
                   location_name='station2',
                   dataset_id='run001')

# save the data to disk in netCDF format
save_cube(cube1)
save_cube(cube2)

cube_list = [cube1, cube2]

# plot on existing axes
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# plot all cubes, label each time series by the location_name attribute
plot_timeseries(ax, cube_list, label_attr='location_name')
plt.show()

# a method to generate image filenames
print(generate_img_filename(cube_list))

# single command to create a new figure, plot and save the image
save_timeseries_figure(cube_list, label_attr='location_name')
