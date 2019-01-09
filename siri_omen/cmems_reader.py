"""
Tools for importing CMEMS observational data.
"""
import numpy
import glob
import datetime
from collections import defaultdict
import iris
from iris.experimental.equalise_cubes import equalise_attributes
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

    location_name = cube.attributes['site_code']
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
                            outputdir=None, verbose=False):
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
    """
    all_cubes = defaultdict(iris.cube.CubeList)

    file_list = sorted(glob.glob(search_pattern))

    # check all matching files, try to read given variable
    for f in file_list:
        try:
            cube_list = read_cmems_file(f, standard_name, start_time, end_time,
                                        verbose=verbose)
            for cube in cube_list:
                location_name = cube.attributes['site_code']
                depth_str = utility.get_depth_sring(cube)
                key = '-'.join([location_name, depth_str])
                cube.attributes['location_name'] = location_name
                cube.attributes['dataset_id'] = dataset_id
                utility.assert_cube_metadata(cube)
                utility.assert_cube_valid_data(cube)
                all_cubes[key].append(cube)
        except AssertionError:
            pass
        except ValueError as e:
            print('Reading file {:} failed.'.format(f))
            print(e)

    # concatenate time dimension in all found files
    for k in all_cubes:
        cube_list = all_cubes[k]
        # concatenate times
        equalise_attributes(cube_list)
        try:
            cube = cube_list.concatenate_cube()
        except Exception as e:
            print('Concatenation failed for {:}'.format(k))
            print(e)

        cube = utility.drop_singleton_dims(cube)
        utility.save_cube(cube, root_dir=outputdir)
