"""
Compute error metrics from time series
"""
from siri_omen import *

obs_id = 'cmems-nrt'
mod_id = 'run002'

var_list = ['slev', 'temp']

start_time = datetime.datetime(2016, 7, 1)
end_time = datetime.datetime(2018, 7, 1)

data_id_list = [obs_id, mod_id]

for var in var_list:
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

    # save_taylor_diagram(cube_pairs, label_attr='location_name')
    save_taylor_target_diagram(cube_pairs, label_attr='location_name',
                               target_datalim=2.2)

