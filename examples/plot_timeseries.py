"""
Make time series plots.
"""
from siri_omen import *

data_id_list = [
    'cmems-nrt',
    'run001',
    'run002',
]
var_list = ['slev', 'temp']

start_time = datetime.datetime(2016, 6, 1)
end_time = datetime.datetime(2018, 7, 1)

for var in var_list:
    dataset_list = []
    for data_id in data_id_list:
        d = read_dataset(data_id, 'timeseries', var)
        dataset_list.append(d)

    # find pairs
    pairs = find_station_pairs(*dataset_list)

    for key in pairs:
        try:
            cube_list = []
            for data_id in data_id_list:
                if data_id in pairs[key]:
                    cube = pairs[key][data_id]
                    cube_list.append(cube)
            data_id_str = '-'.join(data_id_list)
            datatype = 'timeseries'
            outdir = os.path.join('plots', data_id_str, datatype, var)
            save_timeseries_figure(cube_list,
                                output_dir=outdir,
                                alpha=0.7,
                                start_time=start_time,
                                end_time=end_time,
                                time_extent='intersection')
        except Exception:
            pass
