"""
Tools for extracting time series data
"""
from . import utility
import iris
import glob
from .nemo_reader import TimeSeriesExtractor


def parse_target_coordinates(obs_file):
    """
    Read observation netCDF file and return target coordinates and metadata.

    :arg str obs_file: Observation netCDF filename from which target
        coordinates and additional metadata will be read from.
    :returns: lon, lat, station_name, nemo_var, var
    """
    # TODO generalize to 3D data, parse z value as well.
    obs_list = iris.load(obs_file)
    assert len(obs_list) > 0, 'Observation file not found {:}'.format(obs_list)
    obs = obs_list[0]
    lat = obs.coord('latitude').points[0]
    lon = obs.coord('longitude').points[0]
    depth = obs.coord('depth').points[0]
    location_name = obs.attributes['location_name']
    var = obs.standard_name
    nemo_var = utility.map_var_name.get(var, var)
    z = -depth
    return lon, lat, z, location_name, nemo_var, var


def extract_from_obs(run_id, ts_extractor, obs_file):
    """
    Extract time series given an example observation file.

    :arg str run_id: Data set identifier for the simulation, e.g. "testrun01"
    :arg ts_extractor: TimeSeriesExtractor object
    :arg str obs_file: Observation netCDF filename from which target
        coordinates and additional metadata will be read from.
    :returns: an iris Cube object
    """
    # find coordinate arrays
    lon, lat, z, loc_name, nemo_var, var = parse_target_coordinates(obs_file)

    cube = ts_extractor.extract(nemo_var,
                                lon=lon, lat=lat, z=z,
                                location_name=loc_name,
                                dataset_id=run_id)
    return cube


def extract_timeseries_from_obs(run_id, source_file_pattern,
                                obs_search_pattern, outputdir=None):
    """
    Extract a time series for all matching observation netCDF files
    """
    obs_files = sorted(glob.glob(obs_search_pattern))

    if outputdir is None:
        outputdir = run_id
    utility.create_directory(outputdir)
    # construct extractor
    ts_extractor = TimeSeriesExtractor(source_file_pattern)

    for obs_file in obs_files:
        cube = extract_from_obs(run_id, ts_extractor, obs_file)
        utility.save_cube(cube, root_dir=outputdir)
