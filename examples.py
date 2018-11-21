"""
iris NEMO examples
"""
import iris_nemo_reader as nr
import iris_plotting as ip
import matplotlib.pyplot as plt

# extract time series with correct location and dataset metadata
te = nr.TimeSeriesExtractor('NORDIC_1h_20180912_20181029_grid_T_201809*.nc')
cube1 = te.extract('sea_water_practical_salinity',
                   lon=18.9377, lat=60.5332, z=-20.0,
                   location_name='station1',
                   dataset_name='observation')

cube2 = te.extract('sea_water_practical_salinity',
                   lon=19.5797, lat=61.0833, z=-30.0,
                   location_name='station2',
                   dataset_name='run001')

cube_list = [cube1, cube2]

# plot on existing axes
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

print(ip.generate_img_filename(cube_list))

ip.plot_timeseries(ax, cube_list)
plt.show()

# create a new figure, plot and save image
ip.save_timeseries_figure(cube_list)

