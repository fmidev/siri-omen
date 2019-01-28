"""
Make vertical profile plots.
"""
from siri_omen import *

xlim_var = {
    'temp': [0, 25],
    'psal': ([0, 15], [0, 35]),
}

data_id_list = [
    'ices-ctd',
    'run001',
    'run002',
]

var_list = ['temp', 'psal']

for var in var_list:

    dataset_list = []
    for data_id in data_id_list:
        d = read_dataset(data_id, 'profile', var)
        dataset_list.append(d)

    # find pairs
    pairs = find_station_pairs(*dataset_list, time_threshold=300.)

    for key in pairs:
        cube_list = []
        for data_id in data_id_list:
            if data_id in pairs[key]:
                cube = pairs[key][data_id]
                cube_list.append(cube)
        save_profile_figure(cube_list,
                            xlim=xlim_var.get(var),
                            alpha=0.7)
