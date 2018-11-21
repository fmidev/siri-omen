"""
iris NEMO examples
"""
import iris_nemo_reader as nr
import iris_plotting as ip
import matplotlib.pyplot as plt
import glob
import os

import iris


# map standard_name attributes to values used in NEMO output files
map_var_name = {
    'water_surface_height_above_reference_datum': 'sea_surface_height_above_geoid',

}

short_name = {
    'water_surface_height_above_reference_datum': 'ssh',
    'sea_surface_height_above_geoid': 'ssh',
}

# extract time series with correct location and dataset metadata
#te = nr.TimeSeriesExtractor('NORDIC_1h_20180901_20181029_SURF_grid_T_2018091*.nc')
#cube1 = te.extract('sea_surface_height_above_geoid',
                   #lon=18.9377, lat=60.5332,
                   #location_name='station1',
                   #dataset_name='observation')
#print(cube1)


def get_cube_datetime(cube, index):
    time = cube.coord('time')
    return time.units.num2date(time.points[index])


def create_directory(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('file with same name exists', path)
    else:
        os.makedirs(path)
    return path


def gen_filename(cube, data_id, root_dir='obs'):
    prefix = 'ts'
    try:
        station = cube.attributes['site_code']
    except KeyError as e:
        station = cube.attributes['location_name']
    var = cube.standard_name
    var = short_name[var]
    start_date = get_cube_datetime(cube, 0)
    end_date = get_cube_datetime(cube, -1)
    date_str = '_'.join([d.strftime('%Y-%m-%d')
                         for d in [start_date, end_date]])
    fname = '_'.join([prefix, station, data_id, var, date_str]) + '.nc'
    fname = os.path.join(root_dir, fname)
    return fname


def save_cube(cube, data_id, root_dir):
    fname = gen_filename(cube, data_id, root_dir=outputdir)
    print('Saving to {:}'.format(fname))
    iris.save(cube, fname)


def extract_target_coordinates(obs_file):
    obs_list = iris.load(obs_file)
    assert len(obs_list) > 0, 'Observation file not found {:}'.format(obs_list)
    obs = obs_list[0]
    lat = obs.coord('latitude').points[0]
    lon = obs.coord('longitude').points[0]
    station_name = obs.attributes['site_code']
    var = obs.standard_name
    nemo_var = map_var_name.get(var, var)
    return lon, lat, station_name, nemo_var, var


def extract_from_obs(run_id, te, obs_file):
    # find coordinate arrays
    lon, lat, station_name, nemo_var, var = extract_target_coordinates(obs_file)

    cube = te.extract(nemo_var,
                      lon=lon, lat=lat,
                      location_name=station_name,
                      dataset_name=run_id)
    return cube


run_id = 'nemo4-run013'
outputdir = run_id

obs_search_pattern = 'obs/ts_*.nc'
obs_files = sorted(glob.glob(obs_search_pattern))
source_file_pattern = 'NORDIC_1h_20180901_20181029_SURF_grid_T_2018091*.nc'

create_directory(outputdir)
# construct extractor
te = nr.TimeSeriesExtractor(source_file_pattern)

for obs_file in obs_files:
    cube = extract_from_obs(run_id, te, obs_file)
    save_cube(cube, run_id, root_dir=outputdir)
