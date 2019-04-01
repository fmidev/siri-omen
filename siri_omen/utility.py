"""
Misc utility functions
"""
import os
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import numpy
from collections import OrderedDict
import datetime
import pytz
from . import statistics
import gsw  # TEOS-10

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
    'tke': 'specific_turbulent_kinetic_energy_of_sea_water',
    'eps': 'specific_turbulent_kinetic_energy_dissipation_in_sea_water',
    'vdiff': 'ocean_vertical_heat_diffusivity',
    'vvisc': 'ocean_vertical_momentum_diffusivity',
    'u': 'sea_water_x_velocity',
    'v': 'sea_water_y_velocity',
    'w': 'upward_sea_water_velocity',
    'icearea': 'sea_ice_area',
    'iceextent': 'sea_ice_extent',
    'icevol': 'sea_ice_volume',
}

# reverse map: standard_name -> short_name
map_var_short_name = dict((t[1], t[0]) for t in map_var_standard_name.items())

# map standard name to known synonyms
standard_name_synonyms = {
    'water_surface_height_above_reference_datum':
        'sea_surface_height_above_geoid',
    'sea_water_temperature': 'sea_water_potential_temperature',
}

map_short_datatype = {
    'timeseries': 'ts',
    'profile': 'prof',
    'timeprofile': 'tprof',
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


def check_cube_overlap(one, two):
    st1 = get_cube_datetime(one, 0)
    et1 = get_cube_datetime(one, -1)
    st2 = get_cube_datetime(two, 0)
    et2 = get_cube_datetime(two, -1)
    overlap = st1 < et2 and st2 < et1
    return overlap


def constrain_cube_time(cube, start_time=None, end_time=None):
    """
    Constrain time axis between start_time and end_time

    :kwarg datetime start_time: first time stamp to be included
    :kwarg datetime end_time: last time stamp to be included
    :returns: an iris Cube instance
    :raises: AssertionError if requested time period out of range
    """
    if start_time is None:
        start_time = get_cube_datetime(cube, 0)
    if end_time is None:
        end_time = get_cube_datetime(cube, -1)
    # convert to float in cube units
    time_coord = cube.coord('time')
    st = time_coord.units.date2num(start_time)
    et = time_coord.units.date2num(end_time)
    assert et >= time_coord.points[0], \
        'No overlapping time period found. end_time before first time stamp.'
    assert st <= time_coord.points[-1], \
        'No overlapping time period found. start_time after last time stamp.'
    time_constrain = iris.Constraint(
        coord_values={'time': lambda t: st <= t.point <= et})
    new_cube = cube.extract(time_constrain)
    return new_cube


def get_cube_datatype(cube):
    """
    Detect cube datatype.

    Supported datatypes are:
    point       - ()
    timeseries  - (time)
    profile     - (depth)
    timeprofile - (time, depth)
    """
    coords = [c.name() for c in cube.coords()]
    has_depth = 'depth' in coords
    has_time = 'time' in coords
    if has_depth:
        ndepth = len(cube.coord('depth').points)
    if has_time:
        ntime = len(cube.coord('time').points)
    if (has_time and ntime > 1) and (has_depth and ndepth == 1):
        datatype = 'timeseries'
    elif (has_time and ntime == 1) and (has_depth and ndepth > 1):
        datatype = 'profile'
    elif (has_time and ntime > 1) and (has_depth and ndepth > 1):
        datatype = 'timeprofile'
    else:
        print(cube)
        print('has time : {:} n={:}'.format(
            has_time, ntime if has_time else None))
        print('has depth: {:} n={:}'.format(
            has_depth, ndepth if has_depth else None))
        raise NotImplementedError('Unknown cube data type')
    return datatype


def drop_singleton_dims(cube):
    """
    Extract all coordinates that have only one value.
    """
    shape = cube.shape
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
    datatype = get_cube_datatype(cube)
    prefix = map_short_datatype[datatype]

    location_name = cube.attributes['location_name']
    dataset_id = cube.attributes['dataset_id']
    var = cube.standard_name
    var = map_var_short_name[var]
    start_time = get_cube_datetime(cube, 0)
    end_time = get_cube_datetime(cube, -1)
    ntime = len(cube.coord('time').points)
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


def get_common_time_overlap(cube_list, mode='union'):
    """
    Find a common overlapping time interval of the cubes.

    :arg cube_list: list of cubes
    :arg mode: either 'union' or 'intersection'. If 'intersection' will return
    the time interval in which all cubes have data. If 'union' will find the
    time span that contains all the data.
    """

    st_op = min if mode == 'union' else max
    et_op = max if mode == 'union' else min
    start_time = st_op([get_cube_datetime(c, 0) for c in cube_list])
    end_time = et_op([get_cube_datetime(c, -1) for c in cube_list])
    assert end_time > start_time, 'Could not find overlapping time stamps'
    return start_time, end_time


def generate_img_filename(cube_list, prefix=None, loc_str=None, root_dir=None,
                          start_time=None, end_time=None):
    """
    Generate a canonical name for a vertical profile image file.
    """
    datatype = get_cube_datatype(cube_list[0])
    if prefix is None:
        prefix = map_short_datatype[datatype]

    var_list = [map_var_short_name.get(c.standard_name, c.standard_name)
                for c in cube_list]
    var_list = sorted(unique(var_list))
    var_str = '-'.join(var_list)

    if loc_str is None:
        loc_list = [map_var_short_name.get(c.attributes['location_name'],
                                           c.attributes['location_name'])
                    for c in cube_list]
        loc_list = sorted(unique(loc_list))
        loc_str = '-'.join(loc_list)

    if datatype in ['timeseries', 'timeprofile']:
        if start_time is None or end_time is None:
            start_time, end_time = get_common_time_overlap(cube_list, 'union')
        date_str = '_'.join(
            [d.strftime('%Y-%m-%d') for d in [start_time, end_time]])
    else:
        start_time = min([get_cube_datetime(c, 0) for c in cube_list])
        date_str = start_time.strftime('%Y-%m-%d')

    imgfile = '_'.join((prefix, loc_str, var_str, date_str))
    imgfile += '.png'

    if root_dir is None:
        id_list = [c.attributes['dataset_id'] for c in cube_list]
        id_list = sorted(unique(id_list))
        data_id_str = '-'.join(id_list)
        root_dir = os.path.join('plots', data_id_str, datatype, var_str)

    imgfile = os.path.join(root_dir, imgfile)

    return imgfile


def load_cube(input_file, var):
    """
    Load netcdf file to a cube object

    :arg str input_file: netcdf file name
    :arg str var: standard_name of the variable to read. Alternatively can be
        a shortname, e.g. 'temp' or 'psal'
    """
    # if short name convert to standard_name
    _var = map_var_standard_name.get(var, var)

    cube_list = iris.load(input_file, _var)
    assert len(cube_list) > 0, 'Field "{:}" not found in {:}'.format(
        _var, input_file)
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


def align_cubes(first, second):
    """
    Interpolate cubes on the same grid.

    Data in second cube will be interpolated on the grid of the first.
    """
    o = first
    # make deep copy as cubes will be modified
    m = second.copy()

    assert len(o.data.shape) == 1, 'only 1D cubes are supported'
    assert len(m.data.shape) == 1, 'only 1D cubes are supported'

    # find non-scalar coordinate
    coords = [c.name() for c in o.coords() if len(c.points) > 1]
    coord_name = coords[0]

    # convert model time to obs time
    m_time_coord = m.coord(coord_name)
    o_time_coord = o.coord(coord_name)
    m_time_coord.convert_units(o_time_coord.units)

    scheme = iris.analysis.Linear(extrapolation_mode='mask')
    m2 = m.interpolate([(coord_name, o_time_coord.points)], scheme)

    return m2


def concatenate_cubes(cube_list):
    """
    Concatenate multiple cubes into one.

    Variables must be compatible, e.g. cubes must contain non-overlapping and
    increasing time stamps.
    """
    list = iris.cube.CubeList(cube_list)
    equalise_attributes(list)
    cube = list.concatenate_cube()
    return cube


def compute_cube_statistics(reference, predicted):

    predicted_alinged = align_cubes(reference, predicted)

    r = reference.data
    p = predicted_alinged.data
    return statistics.compute_statistics(r, p)


def cube_cell_thicknesses(cube, return_dictionary=False):
    """
    calculates thicknesses for each depth on the cube.
    as defaults returns it as an array,
    if return_dictionary is set to True, returns dictionary,
    with depth value as key for each thickness.
    This could be used to get the thickness by giving the value from
    the depth coordinate.
    """
    cell_thicknesses = cube.coord('depth').bounds[:, 1]\
        - cube.coord('depth').bounds[:, 0]
    if return_dictionary:
        thickness_dictionary = {}
        for d, t in zip(cube.coord('depth').points, cell_thicknesses):
            thickness_dictionary[d] = t
        return thickness_dictionary
    else:
        return cell_thicknesses


def cube_volumes(cube):
    """
    calculates volumes for each cell based on
    lat, lon and depth axis.
    returns a cube with same dimensions than 'cube',
    with each cell having it's volume.
    """
    # make a cube
    volumes = cube.copy()
    volumes.rename('volume')
    volumes.units = 'm^3'
    if 'invalid_units' in volumes.attributes.keys():
        volumes.attributes.pop('invalid_units')  # cleanup
    volumes.attributes['description'] = \
        "cell volume"
    volumes.coord('depth').convert_units('m')  # should be meters
    volumes.data.data[:] = iris.analysis.cartography.area_weights(cube)
    # data.data as we want to keep the original mask.

    cell_thickness = cube_cell_thicknesses(volumes)
    depth_coord = volumes.coord_dims('depth')[0]
    # tells which axis is depth
    new_shape = numpy.array(volumes.shape)
    # new shape has same dimensions than area.
    new_shape[:] = 1
    # dimensions stay same, but each has just one entry.
    new_shape[depth_coord] = volumes.shape[depth_coord]
    # depth axis is real lenght
    volumes.data.data[:] = volumes.data.data*cell_thickness.reshape(new_shape)
    return volumes


def cube_pressure(cube):
    """
    calculates the pressure for each cell based on
    lat and depth axis. depth must be metres.
    returns a cube with same dimensions than 'cube',
    with each cell having it's pressure.
    """
    # make a cube
    pressure = cube.copy()
    pressure.rename('pressure')
    pressure.units = 'dbar'
    # First, lets form a depth axis we can multiply with:
    depth_index = cube.coord_dims('depth')[0]
    depth_coord = numpy.array(cube.coord('depth').points)
    # depth_index is the index of depth axis.
    shape_of_depth = numpy.array(cube.shape)
    shape_of_depth[:] = 1
    shape_of_depth[depth_index] = cube.shape[depth_index]
    # now weh have shape of same form than data,
    # with depth axis only longer than 1 cell.
    depth_grid = numpy.ones(cube.shape)
    depth_grid = depth_grid*depth_coord.reshape(shape_of_depth)
    # now depth is of same format than cube data
    # same should be done to latitude
    # (as TEOS-10 p_to_z uses lat and depth)

    # First, lets form a  atitudexis we can multiply with:
    latitude_index = cube.coord_dims('latitude')[0]
    latitude_coord = numpy.array(cube.coord('latitude').points)
    # latitude_index is the index of latitude axis.
    shape_of_latitude = numpy.array(cube.shape)
    shape_of_latitude[:] = 1
    shape_of_latitude[latitude_index] = cube.shape[latitude_index]
    # now we have shape of same form than data,
    # with latitude axis only longer than 1 cell.
    latitude_grid = numpy.ones(cube.shape)
    latitude_grid = latitude_grid*latitude_coord.reshape(shape_of_latitude)

    pressure.data = gsw.p_from_z(-1.0*depth_grid, latitude_grid)
    # depth is multiplied by -1.0, as this wants height.
    # return value is numpy array of same size than input cells
    return pressure


def cube_density(salinity, conservative_temperature, pressure=None):
    """
    calculates the density for each cell based on
    salinity and conservative temperature and pressure.
    Assumes all inputs to be same shape cubes
    if pressure is missing, calculates it from the others.
    
    Assumes salinity to be absolute salinity, and
    Temperature to be the conservative temperature.
    returns a cube with same dimensions than 'cube',
    with each cell having it's pressure.
    """
    # If no pressure field is provided, let's create it:
    if pressure is None:
        pressure = cube_pressure(salinity)

    # make a cube
    density = salinity.copy()
    density.rename('density')
    density.units = 'kg m^-3'

    density.data = gsw.rho(salinity.data, conservative_temperature.data, pressure.data)

    return density


def cube_heat_content(salinity, temperature):
    """
    calculates the heat content for each cell on the cube
    salinity must be absolute salinity
    temperature can be potential temperature or conservative temperature

    returns a cube with the content.
    """
    heat_content = temperature.copy()  # want to edit a separate entity.
    # todo: check to ensure temperature in is either potential,
    # or conservative temperature.

    heat_content.convert_units('degC')
    # convert to Conservative Temperature
    if heat_content.name() in ['potential_temperature']:
        heat_content.data = gsw.CT_from_pt(salinity.data, heat_content.data)
    density = cube_density(salinity, heat_content).data
    volumes = cube_volumes(salinity).data
    # either salinity or heat_content is fine above.
    # Now we should have all that is needed, let's combine:
    c0p = 3991.86795711963  # J/(kg K)
    heat_content.convert_units('K')
    heat_content.data = volumes*density*heat_content.data*c0p
    # unit = m^3 * kg/M^3 * K * J/(kg K) = J
    # from TEOS-10 manual:
    # Calculation and use of the thermodynamics properties of seawater
    #
    # conservative_heat_content(S,t,p) = h0/c0p
    # and c0p = 3991.867 957 119 63 J/(kgK)
    heat_content.rename('heat_content')
    heat_content.units = "J"

    return heat_content


def abs_sal_from_pract_sal(p_salinity):
    """
    Takes in a cube of practical salinity.
    (ignores teh given units)
    returns a cube with absolute Salinity,
    units fixed.
    """
    # Create a pressure field. 
    pressure = cube_pressure(p_salinity)
    # create latitude and longitude axis
    latitudes_simple = p_salinity.coords('latitude')[0].points
    longitudes_simple = p_salinity.coords('longitude')[0].points

    latitude_index = p_salinity.coord_dims('latitude')[0]
    shape_of_cube = numpy.array(p_salinity.shape)
    shape_of_cube[:] = 1
    shape_of_cube[latitude_index] = p_salinity.shape[latitude_index]
    latitudes = numpy.ones(shape_of_cube)*\
                latitudes_simple.reshape(shape_of_cube)


    longitude_index = p_salinity.coord_dims('longitude')[0]
    shape_of_cube = numpy.array(p_salinity.shape)
    shape_of_cube[:] = 1
    shape_of_cube[longitude_index] = p_salinity.shape[longitude_index]
    longitudes = numpy.ones(shape_of_cube)*\
                longitudes_simple.reshape(shape_of_cube)

    # make a cube
    a_salinity= p_salinity.copy()
    a_salinity.rename('sea_water_absolute_salnity')
    a_salinity.units = 'g kg-1'

    a_salinity.data = gsw.SA_from_SP(p_salinity.data,
                                     pressure.data,
                                     longitudes,
                                     latitudes)

    return a_salinity
