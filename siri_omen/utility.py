"""
Misc utility functions
"""
import os
import iris
import numpy
from collections import OrderedDict
import datetime
import pytz
import dateutil.parser


# standard names for all used variables
# based on CF standard and oceanSITES short variable names
map_var_standard_name = {
    'airt': 'air_temperature',
    'caph': 'air_pressure',
    'depth': 'depth',
    'hcsp': 'sea_water_speed',
    'pres': 'sea_water_pressure',
    'psal': 'sea_water_practical_salinity',
    'temp': 'sea_water_temperature',
    'ucur': 'eastward_sea_water_velocity',
    'uwnd': 'eastward_wind',
    'vcur': 'northward_sea_water_velocity',
    'vwnd': 'northward_wind',
    'wdir': 'wind_to_direction',
    'wspd': 'wind_speed',
    'slev': 'water_surface_height_above_reference_datum',
    }

# reverse map: standard_name -> short_name
map_var_short_name = dict((t[1], t[0]) for t in map_var_standard_name.items())

# map standard name to known synonyms
standard_name_synonyms = {
    'water_surface_height_above_reference_datum': 'sea_surface_height_above_geoid',
    'sea_water_temperature': 'sea_water_potential_temperature',
}


epoch = datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)
epoch_unit_str = 'seconds since 1970-01-01 00:00:00-00'


def datetime_to_epoch(t):
    """
    Convert python datetime object to epoch time stamp.
    """
    return (t - epoch).total_seconds()


def epoch_to_datetime(t):
    """
    Convert python datetime object to epoch time stamp.
    """
    return epoch + datetime.timedelta(seconds=t)


def unique(input_list):
    """
    Returns unique elements in a list
    """
    return list(OrderedDict.fromkeys(input_list))


def get_cube_datetime(cube, index):
    time = cube.coord('time')
    return time.units.num2date(time.points[index])


def get_depth_sring(cube):
    depth = cube.coord('depth').points.mean()
    depth_str = 'd{:.2f}m'.format(depth)
    return depth_str


def get_time_summary(cube):
    start_time = get_cube_datetime(cube, 0)
    end_time = get_cube_datetime(cube, -1)
    ntime = len(cube.coord('time').points)
    out = 'Time: {:} -> {:}, {:} points'.format(start_time,
                                                end_time, ntime)
    return out


def create_directory(path):
    """
    Create directory in the file system.

    :arg str path: directory path, full or relative
    :raises: IOError if a file with the same name already exists.
    """
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError('file with same name exists', path)
    else:
        os.makedirs(path)
    return path


def assert_cube_metadata(cube):
    """
    Asserts that cube has all the required metadata.
    """
    attributes = [
        'location_name',
        'dataset_id'
    ]
    for a in attributes:
        msg = 'Cube does not have "{:}" attribute'.format(a)
        assert a in cube.attributes, msg


def assert_cube_valid_data(cube):
    """
    Asserts that cube contains non nan/inf/masked data.
    """
    if numpy.ma.is_masked(cube.data):
        assert not cube.data.mask.all(), 'All data is masked'
    assert numpy.isfinite(cube.data).any(), 'All data is nan or inf'


def constrain_cube_time(cube, start_time=None, end_time=None):
    """
    Constrain time axis between start_time and end_time

    :kwarg datetime start_time: first time stamp to be included
    :kwarg datetime end_time: last time stamp to be included
    :returns: an iris Cube instance
    :raises: AssertionError if requested time period out of range
    """
    time = cube.coord('time')
    assert time in cube.dim_coords, 'Time is not a DimCoord instance'
    time_dim_index = cube.coord_dims('time')
    assert len(time_dim_index) == 1
    time_dim_index = time_dim_index[0]

    time_array = time.points
    if start_time is not None:
        t_st = time.units.date2num(start_time)
    else:
        t_st = time_array[0]
    if end_time is not None:
        t_et = time.units.date2num(end_time)
    else:
        t_et = time_array[-1]

    tix = (time_array <= t_et) * (time_array >= t_st)
    assert numpy.any(tix), 'No suitable time stamps found'

    ndims = len(cube.shape)
    if ndims == 1:
        slice_obj = tix
    else:
        slice_obj = [slice(None, None, None)] * ndims
        slice_obj[time_dim_index] = tix

    # slice me
    new_cube = cube[slice_obj]
    return new_cube


def drop_singleton_dims(cube):
    """
    Extract all coordinates that have only one value.
    """
    shape = cube.data.shape
    extract = [0] * len(shape)
    for i, l in enumerate(shape):
        if l > 1:
            extract[i] = slice(l)
    new_cube = cube[tuple(extract)]
    return new_cube


def gen_filename(cube, root_dir='obs'):
    """
    Generate a canonical file name for a Cube

    File name is generated from the cube metadata.
    """
    assert_cube_metadata(cube)
    ndepth = 1
    coords = [c.name() for c in cube.coords()]
    if 'depth' in coords:
        ndepth = len(cube.coord('depth').points)
    ntime = len(cube.coord('time').points)
    if ndepth == 1:
        datatype = 'timeseries'
    elif ndepth > 1 and ntime == 1:
        datatype = 'profile'
    elif ndepth > 1 and ntime > 1:
        datatype = 'timeprofile'
    else:
        raise NotImplementedError('Unknown cube data type')

    type_abbrev = {
        'timeseries': 'ts',
        'profile': 'prof',
        'timeprofile': 'tprof',
    }
    prefix = 'ts' if datatype == 'timeseries' else 'vprof'

    location_name = cube.attributes['location_name']
    dataset_id = cube.attributes['dataset_id']
    var = cube.standard_name
    var = map_var_short_name[var]
    start_time = get_cube_datetime(cube, 0)
    end_time = get_cube_datetime(cube, -1)
    if ntime == 1:
        date_str = start_time.strftime('%Y-%m-%d')
    else:
        date_str = '_'.join([d.strftime('%Y-%m-%d')
                             for d in [start_time, end_time]])
    if datatype in ['profile', 'timeprofile']:
        parts = [prefix, location_name, dataset_id, var, date_str]
    else:
        depth_str = get_depth_sring(cube)
        parts = [prefix, location_name, depth_str, dataset_id, var, date_str]
    fname = '_'.join(parts) + '.nc'
    dir = root_dir if root_dir is not None else ''
    dir = os.path.join(dataset_id, dir, datatype, location_name, var)
    create_directory(dir)
    fname = os.path.join(dir, fname)
    return fname


def load_cube(input_file, var):
    cube_list = iris.load(input_file, var)
    assert len(cube_list) > 0, 'Field "{:}" not found in {:}'.format(
        var, input_file)
    assert len(cube_list) == 1, 'Multiple files found'
    cube = cube_list[0]
    return cube


def save_cube(cube, root_dir=None, fname=None):
    """
    Saves a cube in to disk.
    """
    if fname is None:
        fname = gen_filename(cube, root_dir=root_dir)
    print('Saving to {:}'.format(fname))
    iris.save(cube, fname)
