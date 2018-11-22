"""
Example of batch plotting time series
"""
from siri_omen import *
from collections import defaultdict


def read_timeseries_files(search_pattern, var):
    """
    Reads timeseries from netCDF files.

    Returns a dict of cubes.
    """
    output = {}

    def gen_unique_key(cube):
        location_name = cube.attributes['location_name']
        depth = cube.coord('depth').points[0]
        depth_str = 'd{:g}m'.format(depth)
        key = '-'.join([location_name, depth_str])
        return key

    file_list = sorted(glob.glob(search_pattern))
    assert len(file_list) > 0, 'No files found {:}'.format(search_pattern)
    for f in file_list:
        try:
            cube_list = iris.load(f, var)
            assert len(cube_list) > 0
            cube = cube_list[0]
            assert_cube_valid_data(cube)
            key = gen_unique_key(cube)
            output[key] = cube
        except AssertionError as e:
            pass


    return output


def get_time_overlap(cube_list):
    max_start_time = max([get_cube_datetime(c, 0) for c in cube_list])
    min_end_time = min([get_cube_datetime(c, -1) for c in cube_list])
    overlap = (min_end_time - max_start_time)
    return overlap


def check_time_overlap(cube_list, min_overlap_days=1.0):
    overlap = get_time_overlap(cube_list)
    overlap_days = overlap.total_seconds()/(24*3600.)
    return overlap_days >= min_overlap_days


def batch_plot_timeseries(cube_dict_list, imgdir='plots'):
    # find pairs
    keys = [set(d.keys()) for d in cube_dict_list]
    common_keys = set.intersection(*keys)

    paired_cubes = defaultdict(list)
    for key in common_keys:
        cube_list = []
        for s in cube_dict_list:
            cube_list.append(s[key])
        if check_time_overlap(cube_list):
            paired_cubes[key] = cube_list

    # make plots

    offset = datetime.timedelta(days=1)
    start_time = datetime.datetime(2018, 9, 1) - offset
    end_time = datetime.datetime(2018, 10, 30) + offset

    time_lim = [start_time, end_time]
    for k in paired_cubes:
        cube_list = paired_cubes[k]
        save_timeseries_figure(cube_list, label_attr='dataset_id',
                               output_dir=imgdir, time_lim=time_lim)


# TODO find a better way for mapping variable names
obs_standard_names = {
    'ssh': 'water_surface_height_above_reference_datum',
    'salt': 'sea_water_practical_salinity',
    'temp': 'sea_water_temperature',
}

model_standard_names = {
    'ssh': 'sea_surface_height_above_geoid',
    'salt': 'sea_water_practical_salinity',
    'temp': 'sea_water_potential_temperature',
}


def model_obs_comparison(var_short_name):

    search_pattern = 'obs/ts_*_{var:}_*.nc'.format(var=var_short_name)
    var = obs_standard_names[var_short_name]
    obs_cubes = read_timeseries_files(search_pattern, var)

    search_pattern = 'nemo4-run013/ts_*_{var:}_*.nc'.format(var=var_short_name)
    var = model_standard_names[var_short_name]
    mod_cubes = read_timeseries_files(search_pattern, var)

    img_root_dir = 'plots'
    imgdir = os.path.join(img_root_dir, var_short_name)
    batch_plot_timeseries([obs_cubes, mod_cubes], imgdir=imgdir)

model_obs_comparison('ssh')
model_obs_comparison('temp')
model_obs_comparison('salt')
