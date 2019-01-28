from siri_omen import *


obs_id = 'cmems-nrt'
mod_id = 'run001'

datatype = 'timeseries'
var = 'temp'

start_time = datetime.datetime(2016, 7, 1)
end_time = datetime.datetime(2018, 7, 1)

data_id_list = [obs_id, mod_id]

diag = GeographicPlot()
diag.add_feature('land')
diag.ax.set_prop_cycle(get_point_style_cycler())
diag.add_title(' '.join([datatype, obs_id,
                         map_var_standard_name[var].replace('_', ' ')]))

dataset_list = []
for data_id in data_id_list:
    d = read_dataset(data_id, 'timeseries', var,
                        start_time=start_time, end_time=end_time)
    dataset_list.append(d)

# find pairs
pairs = find_station_pairs(*dataset_list)

cube_pairs = []
for key in pairs:
    o = pairs[key][obs_id]
    m = pairs[key][mod_id]
    cube_pairs.append((o, m))

    lat = o.coord('latitude').points
    lon = o.coord('longitude').points
    location_name = o.attributes['location_name']
    diag.add_station(lon, lat, label=location_name, label_to_legend=True,
                        alpha=0.7)

diag.add_legend()

imgname = 'plot_station_map_{:}_{:}'.format(datatype, var)
print('Saving image {:}'.format(imgname))
plt.savefig(imgname, dpi=200, bbox_inches='tight')
