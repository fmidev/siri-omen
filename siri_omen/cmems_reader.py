"""
Tools for importing CMEMS observational data.
"""
import numpy
import glob
from collections import defaultdict
import iris
from . import utility

__all__ = [
    'read_cmems_file',
    'import_cmems_timeseries',
]


def _get_depth_coord(input_file, var):
    cube = utility.load_cube(input_file, var)
    depth = cube.data.mean(axis=0)
    valid_mask = None
    if numpy.ma.is_masked(depth):
        valid_mask = ~depth.mask
        if numpy.isscalar(valid_mask):
            valid_mask = numpy.ones_like(depth, dtype=bool) * valid_mask
        depth = depth[valid_mask]
    depth_coord = iris.coords.DimCoord(depth,
                                       standard_name=cube.standard_name,
                                       units=cube.units)
    return depth_coord, valid_mask


def _get_location_name(cube):
    attr = cube.attributes

    def read_attr(name):
        if name in attr and attr[name] != '':
            return attr[name]
        return None

    location_name = None
    attr_names = [
        'platform_name',
        'platform_code',
        'site_code',
        'wmo_platform_code',
    ]
    for n in attr_names:
        location_name = read_attr(n)
        if not (location_name is None or location_name.isspace()):
            break
    assert location_name is not None
    assert not location_name.isspace()
    assert location_name != ''
    if location_name[-2:] == 'TG':
        location_name = location_name[:-2]
    return location_name


def read_cmems_file(input_file, var, start_time=None, end_time=None,
                    verbose=False):
    """
    Read CMEMS time series file into a cube object
    """
    if verbose:
        print('Reading file {:}'.format(input_file))
    cube = utility.load_cube(input_file, var)

    # try loading quality flags
    nc_varname = cube.var_name
    qc_cube_list = iris.load(input_file, 'quality flag')
    for c in qc_cube_list:
        if c.var_name == nc_varname + '_QC':
            good_qc_flag = 1
            valid_mask = numpy.min(c.data == good_qc_flag, axis=1)
            assert valid_mask.any(), 'All values flagged bad'
            # filter bad values
            cube = cube[valid_mask, :]

    location_name = _get_location_name(cube)
    assert not location_name.isspace(), \
        'Bad location name "{:}" in {:}'.format(location_name, input_file)

    # add lat,lon coordinate dimensions
    lat = float(cube.attributes['last_latitude_observation'])
    lon = float(cube.attributes['last_longitude_observation'])
    lat_coord = iris.coords.DimCoord(lat, standard_name='latitude',
                                     units='degrees')
    lon_coord = iris.coords.DimCoord(lon, standard_name='longitude',
                                     units='degrees')
    cube.add_aux_coord(lat_coord)
    cube.add_aux_coord(lon_coord)

    depth_coord, valid_mask = _get_depth_coord(input_file, 'depth')
    if valid_mask is not None:
        cube = cube[:, valid_mask]
    cube.add_dim_coord(depth_coord, 1)

    # cut each depth to separate cube
    output = []
    for single_depth in cube.slices('time'):
        output.append(single_depth)

    return output


def import_cmems_timeseries(dataset_id,
                            search_pattern, standard_name,
                            start_time=None, end_time=None,
                            outputdir=None,
                            skip_station_list=None,
                            include_station_list=None,
                            verbose=False):
    """
    Read CMEMS time series multiple netCDF files and stores it to disk.

    :arg str dataset_id: Human readable dataset identifier e.g. 'obs'
    :arg str search_pattern: Search pattern for CMEMS source files, e.g.
        'somedir/file_*.nc'
    :arg str standard_name: variable CF convention standard_name, e.g.
        'water_surface_height_above_reference_datum'
    :kwarg datetime start_time: first time stamp to accept (optional)
    :kwarg datetime end_time: last time stamp to accept (optional)
    :kwarg str outputdir: root directory of the stored files
        (default: dataset_id)
    :kwarg skip_station_list: list of station names that will be skipped.
    :kwarg include_station_list: If provided, read only netcdf files that
        contain one of these names.
    """
    all_cubes = defaultdict(iris.cube.CubeList)

    file_list = sorted(glob.glob(search_pattern))

    # check all matching files, try to read given variable
    for f in file_list:
        if include_station_list is not None:
            # check that f contains at least one of the names
            if not any([s in f for s in include_station_list]):
                continue
        try:
            cube_list = read_cmems_file(f, standard_name, start_time, end_time,
                                        verbose=verbose)
            for cube in cube_list:
                location_name = _get_location_name(cube)
                depth_str = utility.get_depth_string(cube)
                key = '-'.join([location_name, depth_str])
                cube.attributes['location_name'] = location_name
                cube.attributes['dataset_id'] = dataset_id
                utility.assert_cube_metadata(cube)
                utility.assert_cube_valid_data(cube)
                all_cubes[key].append(cube)
        except AssertionError as e:
            print(e)
        except ValueError as e:
            print('Reading file {:} failed.'.format(f))
            print(e)

    # concatenate time dimension in all found files
    for k in all_cubes:
        cube_list = all_cubes[k]
        # concatenate times
        try:
            cube = utility.concatenate_cubes(cube_list)
        except Exception as e:
            print('Concatenation failed for {:}'.format(k))
            print(e)

        cube = utility.drop_singleton_dims(cube)
        location_name = cube.attributes['location_name']
        if skip_station_list and location_name in skip_station_list:
            print('Skipping station {:}'.format(location_name))
            continue
        try:
            cube = utility.constrain_cube_time(cube, start_time, end_time)
            utility.save_cube(cube, root_dir=outputdir)
        except Exception as e:
            print('Could not save cube: {:}'.format(k))
            print(e)
