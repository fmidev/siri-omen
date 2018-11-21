"""
Read NEMO model output files into Iris cube objects.

Examples:

Read 3D field:

cube = load_nemo_output('NORDIC_1h_grid_T_20180915-20180915.nc',
                        'sea_water_practical_salinity')


Extract time series from a list of output files

te = TimeSeriesExtractor('NORDIC_1h_20180912_20181029_grid_T_201809*.nc')
cube = te.extract('sea_water_practical_salinity',
                  lon=18.9377, lat=60.5332, z=-20.0)
"""
import numpy
import iris
from scipy.spatial import cKDTree as KDTree
import netCDF4
import glob
import collections


class NearestNeighborFinder():
    """
    Nearest neighbor search object for NEMO netCDF output files.
    """
    def __init__(self, ncfilename):
        """
        Create new instance.

        :arg str ncfilename: NEMO netCDF file name
        """
        self.filename = ncfilename
        self._build_tree()
        self.data_dim = None

    def _build_tree(self):
        """
        Construct nearest neighbor tree.
        """
        with netCDF4.Dataset(self.filename) as ncf:
            # compute land mask
            self.data_dim = 3 if 'e3t' in ncf.variables else 2
            if self.data_dim == 3:
                # NOTE does not take time-dependent wetting-drying into account
                e = ncf['e3t'][0, :, :, :]
                self.landmask = numpy.all(e.mask, axis=0).T
                # 1D array of all wet points in raveled index
                self.wetmask = numpy.nonzero(~self.landmask.ravel())[0]
                # get coordinates
                self.lon = ncf['nav_lon'][:].T
                self.lat = ncf['nav_lat'][:].T
                depth = ncf['deptht'][:]
                self.z = -depth
                # 1D arrays of all wet points
                self.valid_lon = self.lon.ravel()[self.wetmask]
                self.valid_lat = self.lat.ravel()[self.wetmask]
            else:
                # read a field to get landmask
                for v in ncf.variables:
                    var = ncf[v]
                    if len(var.shape) == 3:
                        # 2D time dependent field
                        self.landmask = numpy.all(var[:].mask, axis=0).T
                        break
                self.wetmask = numpy.nonzero(~self.landmask.ravel())[0]
                # get coordinates
                self.lon = ncf['nav_lon'][:].T
                self.lat = ncf['nav_lat'][:].T
                self.z = 0.0
                # 1D arrays of all wet points
                self.valid_lon = self.lon.ravel()[self.wetmask]
                self.valid_lat = self.lat.ravel()[self.wetmask]

        coords = numpy.vstack((self.valid_lon, self.valid_lat)).T
        self.tree = KDTree(coords)

    def find(self, lon, lat, z):
        """
        Finds nearest neighbor index for point (lon, lat, z)

        :arg lon: longitude coordinate
        :arg lat: latitude coordinate
        :arg z: z coordinate (negative downwards)
        :returns: i, j, k indices of nearest neighbor indices
        """
        dist, index = self.tree.query([lon, lat], k=1)
        index = self.wetmask[index]
        i, j = numpy.unravel_index(index, self.lat.shape)
        if self.data_dim == 3:
            k = numpy.abs(self.z - z).argmin()
        else:
            k = None
        return i, j, k


class TimeSeriesExtractor():
    """
    Extracts time series from NEMO netCDF output files.

    Finds a nearest point for a given (lon, lat, z) query point. Extracts a
    time series from all input files that match the filename pattern.
    Concatenates the time series into a single cube object.
    """
    def __init__(self, ncfilename_pattern):
        """
        :arg ncfilename_pattern: File pattern for input file search. E.g.
            'output_2001*.nc'
        """
        if isinstance(ncfilename_pattern, str):
            self.filename_pattern = ncfilename_pattern
            self.file_list = None
        elif collections.Iterable(ncfilename_pattern):
            self.file_list = list(ncfilename_pattern)
        else:
            raise Exception('Unsupported ncfilename_pattern type')
        self._initialized = False

    def _initialize(self):
        """
        Initializes seach objects.
        """
        if self.file_list is None:
            self._find_files()
        # build search objects for the first file
        # assuming that the rest are using the same grid
        self.nn_finder = NearestNeighborFinder(self.file_list[0])
        self._initialized = True

    def _find_files(self):
        """
        Use glob to search for matching input files.
        """
        file_list = glob.glob(self.filename_pattern)
        assert len(file_list) > 0, \
            'No files found: {:}'.format(self.filename_pattern)
        self.file_list = file_list

    def extract(self, var, lon, lat, z=0.0, location_name=None, dataset_name=None):
        """
        Reads a time series from the source file at the specified location

        :arg lon: longitude coordinate
        :arg lat: latitude coordinate
        :arg z: z coordinate (negative downwards)
        :kwarg location_name: human readable name of the location
            (e.g. station name)
        :kwarg dataset_name: human readable name of the dataset
            (e.g. instrument or model run identifier)
        :returns: iris Cube object of the time series
        """
        if not self._initialized:
            self._initialize()
        i, j, k = self.nn_finder.find(lon, lat, z)
        cube_list = iris.cube.CubeList()
        for filename in self.file_list:
            print('Reading file {:}'.format(filename))
            ncvar = None
            with netCDF4.Dataset(filename) as f:
                for name in ['standard_name', 'long_name']:
                    for vname in f.variables:
                        v = f[vname]
                        if (hasattr(v, 'standard_name')
                                and getattr(v, 'standard_name') == var):
                            # print('Found variable {:}: {:}'.format(var, vname))
                            ncvar = v
                            break
                    if ncvar is not None:
                        break
                assert ncvar is not None, \
                    'Variable {:} not found in {:}'.format(var, self.filename)
                if self.nn_finder.data_dim == 3:
                    values = ncvar[:, k, j, i]
                else:
                    values = ncvar[:, j, i]
                units = ncvar.units
                long_name = ncvar.long_name
                timevar = f['time_centered']
                time_dim = iris.coords.DimCoord(
                    timevar[:],
                    standard_name=timevar.standard_name,
                    units=timevar.units
                )
            # create Cube object
            lon_dim = iris.coords.DimCoord(lon, standard_name='longitude',
                                           units='degrees')
            lat_dim = iris.coords.DimCoord(lat, standard_name='latitude',
                                           units='degrees')
            dep_dim = iris.coords.DimCoord(-z, standard_name='depth',
                                           units='m')
            cube = iris.cube.Cube(values,
                                  standard_name=var, long_name=long_name,
                                  units=units)
            cube.add_dim_coord(time_dim, 0)
            cube.add_aux_coord(dep_dim, None)
            cube.add_aux_coord(lat_dim, None)
            cube.add_aux_coord(lon_dim, None)
            cube_list.append(cube)
        output = cube_list.concatenate_cube()
        if location_name is not None:
            output.attributes['location_name'] = location_name
        if dataset_name is not None:
            output.attributes['dataset_name'] = dataset_name
        return output


def fix_cube_coordinates(cube):
    """
    Fixes NEMO lat,lon coordinates to format that iris supports

    Changes the Cube object in-place.

    :arg cube: a Cube object representing a 2D or 3D NEMO output field.
    """
    # lat,lon coordinates are stored in 2D array which iris does not understand
    # convert coordinates to 1D lat, lon dimensions
    lat = cube.coord('latitude')
    lat = numpy.unique(lat.points.filled(numpy.nan))
    lat_coord = iris.coords.DimCoord(lat, standard_name='latitude',
                                     units='degrees')

    lon = cube.coord('longitude')
    lon = numpy.unique(lon.points.filled(numpy.nan))
    lon_coord = iris.coords.DimCoord(lon, standard_name='longitude',
                                     units='degrees')

    # remove previous coordinates from the cube
    lat_index, lon_index = cube.coord_dims('latitude')
    cube.remove_coord('latitude')
    cube.remove_coord('longitude')
    # there's two coordinates defined with name 'time'
    # remove the latter AuxCoord instance
    for c in cube.coords():
        if isinstance(c, iris.coords.AuxCoord) and c.standard_name == 'time':
            cube.remove_coord(c)

    # add new coordinates to the cube
    # the indices 1,2 must match the data array dims
    cube.add_dim_coord(lat_coord, lat_index)
    cube.add_dim_coord(lon_coord, lon_index)
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()

    # fix vertical coordinate
    has_z_coords = 'deptht' in [c.var_name for c in cube.coords()]
    if has_z_coords:
        c = cube.coord('Vertical T levels')
        dep_array = c.points
        z_coord = iris.coords.DimCoord(dep_array,
                                       standard_name='depth',
                                       units='m')
        z_dim_index = cube.coord_dims(c.long_name)[0]
        cube.remove_coord(c.long_name)
        cube.add_dim_coord(z_coord, z_dim_index)


def load_nemo_output(ncfile, standard_name):
    """
    Load a field identified with standard_name from NEMO output file.

    Replaces 2D lat,lon coordinate arrays with 1D arrays.

    :arg ncfile: netCDF file name to read
    :arg standard_name: CF standard_name of a field to read
    :returns: an iris Cube object
    """
    cube_list = iris.load(ncfile, standard_name)
    assert len(cube_list) > 0, 'No field {:} found in {:}'.format(
        standard_name, ncfile)
    assert len(cube_list) == 1, 'Multiple fields found'
    cube = cube_list[0]
    fix_cube_coordinates(cube)
    return cube
