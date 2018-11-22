"""
Misc utility functions
"""
import os
import iris
import numpy
from collections import OrderedDict


# map standard_name attributes to values used in NEMO output files
map_var_name = {
    'water_surface_height_above_reference_datum': 'sea_surface_height_above_geoid',
    'sea_water_temperature': 'sea_water_potential_temperature',
}

map_var_short_name = {
    'water_surface_height_above_reference_datum': 'ssh',
    'sea_surface_height_above_geoid': 'ssh',
    'sea_water_practical_salinity': 'salt',
    'sea_water_temperature': 'temp',
    'sea_water_potential_temperature': 'temp',
}


def unique(input_list):
    """
    Returns unique elements in a list
    """
    return list(OrderedDict.fromkeys(input_list))


def get_cube_datetime(cube, index):
    time = cube.coord('time')
    return time.units.num2date(time.points[index])


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
        slice_obj = [slice(None, None, None)]*ndims
        slice_obj[time_dim_index] = tix

    # slice me
    new_cube = cube[slice_obj]
    return new_cube


def gen_filename(cube, root_dir='obs'):
    """
    Generate a canonical file name for a Cube

    File name is generated from the cube metadata.
    """
    assert_cube_metadata(cube)
    prefix = 'ts'
    location_name = cube.attributes['location_name']
    dataset_id = cube.attributes['dataset_id']
    depth = cube.coord('depth').points.mean()
    depth_str = 'd{:g}m'.format(depth)
    var = cube.standard_name
    var = map_var_short_name[var]
    start_time = get_cube_datetime(cube, 0)
    end_time = get_cube_datetime(cube, -1)
    date_str = '_'.join([d.strftime('%Y-%m-%d')
                         for d in [start_time, end_time]])
    fname = '_'.join([prefix, location_name, depth_str, dataset_id, var, date_str]) + '.nc'
    if root_dir is not None:
        fname = os.path.join(root_dir, fname)
    return fname


def save_cube(cube, root_dir=None, fname=None):
    """
    Saves a cube in to disk.
    """
    if fname is None:
        fname = gen_filename(cube, root_dir=root_dir)
    print('Saving to {:}'.format(fname))
    iris.save(cube, fname)
