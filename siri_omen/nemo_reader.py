"""
Tools for reading in NEMO output files.

All methods that are NEMO specific should be within this module.
"""
import numpy
from scipy.spatial import cKDTree as KDTree
from iris.experimental.equalise_cubes import equalise_attributes
import cf_units
import iris
import netCDF4
import glob
import collections
from . import utility

map_nemo_standard_name = {
    'sea_water_temperature': 'sea_water_potential_temperature',
    'sea_water_practical_salinity': 'sea_water_practical_salinity',
    'water_surface_height_above_reference_datum':
        'sea_surface_height_above_geoid',
}

# reverse map: standard_name -> short_name
map_nemo_sname_to_standard = dict((t[1], t[0]) for
                                  t in map_nemo_standard_name.items())

# declare correct netcdf variable name for cases where standard_name is
# insufficient to find an unique time series
nemo_ncvar_name = {
    'slev': 'SSH_inst',
}


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
        self.data_dim = None
        self.grid_type = None
        self._build_tree()

    def _build_tree(self):
        """
        Construct nearest neighbor tree.
        """
        def parse_grid_type(ncf):
            """
            Figure out which discretization the file contains, T, U or V

            Reads the description attribute, e.g. "ocean T grid variables"

            returns 't', 'u', or 'v'
            """
            desc = ncf.description
            words = desc.split()
            assert words[0] == 'ocean'
            assert words[2] == 'grid'
            return words[1].lower()

        with netCDF4.Dataset(self.filename) as ncf:
            self.grid_type = parse_grid_type(ncf)
            assert self.grid_type == 't', 'Only T grid is supported currently'
            # compute land mask
            self.data_dim = 3 if 'e3t' in ncf.variables else 2
            if self.data_dim == 3:
                # NOTE does not take time-dependent wetting-drying into account
                e = ncf['e3t'][0, :, :, :]
                self.landmask = numpy.all(e.mask, axis=0).T
                # 1D array of all wet points in raveled index
                self.wetmask = numpy.nonzero(~self.landmask.ravel())[0]
                # get coordinates
                self.lon = ncf['nav_lon'][:]
                self.lat = ncf['nav_lat'][:]
                depth = ncf['deptht'][:]
                self.z = -depth
                # 1D arrays of all wet points
                self.valid_lon = self.lon.T.ravel()[self.wetmask]
                self.valid_lat = self.lat.T.ravel()[self.wetmask]
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
                self.lon = ncf['nav_lon'][:]
                self.lat = ncf['nav_lat'][:]
                self.z = 0.0
                # 1D arrays of all wet points
                self.valid_lon = self.lon.T.ravel()[self.wetmask]
                self.valid_lat = self.lat.T.ravel()[self.wetmask]

        assert len(self.valid_lat) > 0, \
            'No valid points found in {:}'.format(self.filename)
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


class NemoFileReader():
    """
    Object that reads daily/monthly/yearly Nemo output files.

    """
    def __init__(self, ncfilename_pattern, verbose=False):
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
        self.verbose = verbose
        self._initialized = False

    def _initialize(self):
        """
        Initializes seach objects.
        """
        if self.file_list is None:
            self._find_files()
        self._initialized = True

    def _find_files(self):
        """
        Use glob to search for matching input files.
        """
        if self.verbose:
            print('Searching files: {:}'.format(self.filename_pattern))
        file_list = sorted(glob.glob(self.filename_pattern))
        assert len(file_list) > 0, \
            'No files found: {:}'.format(self.filename_pattern)
        self.file_list = file_list


class NemoStationFileReader(NemoFileReader):
    """
    Reads Nemo station files.
    """
    def __init__(self, ncfilename_pattern, dataset_id, verbose=False):
        super().__init__(ncfilename_pattern, verbose=verbose)
        self._initialize()
        self.dataset_id = dataset_id

    def _initialize(self):
        super()._initialize()
        self._find_stations()

    def _find_stations(self):
        self.station_metadata = {}
        for f in self.file_list:
            if self.verbose:
                print('Reading metadata: {:}'.format(f))
            with netCDF4.Dataset(f, 'r') as ncfile:
                name_attr = ncfile.getncattr('name')
                location_name = name_attr.replace('station_', '')
                lat = ncfile['nav_lat'][:][0, 0]
                lon = ncfile['nav_lon'][:][0, 0]
                depth = ncfile['deptht'][:]
                # omit points below bottom
                # FIXME this assumes temp field exists, generalize
                valid_mask = ncfile['temperature'][:, :, 0, 0]
                if isinstance(valid_mask, numpy.ma.MaskedArray):
                    valid_mask = valid_mask.filled(numpy.nan)
                valid_mask = numpy.isfinite(valid_mask)
                valid_mask = numpy.min(valid_mask, axis=0)
                depth = depth[valid_mask]
                if isinstance(depth, numpy.ma.MaskedArray):
                    depth = depth.filled(numpy.nan)
                key = '{:}-lon{:.2f}-lat{:.2f}'.format(location_name, lon, lat)
                if key not in self.station_metadata:
                    meta = {}
                    meta['location_name'] = location_name
                    meta['latitude'] = lat
                    meta['longitude'] = lon
                    meta['depth'] = depth
                    meta['valid_mask'] = valid_mask
                    meta['files'] = []
                    self.station_metadata[key] = meta
                self.station_metadata[key]['files'].append(f)
        if self.verbose:
            print('Found stations:')
            for key in self.station_metadata:
                print(key)
                for f in self.station_metadata[key]['files']:
                    print('    {:}'.format(f))

    def get_station_metadata(self):
        return self.station_metadata

    def get_dataset(self, variable, var_name=None):
        """
        Reads all files to cubes and concatenates them in time
        """
        dataset = {}
        for key in self.station_metadata.keys():
            if self.verbose:
                print('Concatenating data: {:}'.format(key))
            meta = self.station_metadata[key]
            cube_list = iris.cube.CubeList()
            for f in meta['files']:
                if self.verbose:
                    print('Loading {:}'.format(f))
                kw = {}
                if var_name is not None:
                    kw['var_name'] = var_name
                c = load_nemo_output(f, variable, **kw)
                cube_list.append(c)
            equalise_attributes(cube_list)
            cube = cube_list.concatenate_cube()
            cube.attributes['dataset_id'] = self.dataset_id
            cube.attributes.pop('name', None)
            cube.attributes['location_name'] = meta['location_name']
            if isinstance(cube.data, numpy.ma.MaskedArray) \
                    and numpy.all(cube.data.mask):
                # if all data is bad, skip
                continue
            # use correct standard name
            cube.standard_name = map_nemo_sname_to_standard[cube.standard_name]
            cube = utility.drop_singleton_dims(cube)
            dataset[key] = cube
        return dataset


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
        file_list = sorted(glob.glob(self.filename_pattern))
        assert len(file_list) > 0, \
            'No files found: {:}'.format(self.filename_pattern)
        self.file_list = file_list

    def extract(self, var, lon, lat, z=0.0,
                location_name=None, dataset_id=None,
                use_source_coordinates=False):
        """
        Reads a time series from the source file at the specified location

        :arg lon: longitude coordinate
        :arg lat: latitude coordinate
        :arg z: z coordinate (negative downwards)
        :kwarg location_name: human readable name of the location
            (e.g. station name)
        :kwarg dataset_id: human readable name of the dataset
            (e.g. instrument or model run identifier)
        :returns: iris Cube object of the time series
        """
        if not self._initialized:
            self._initialize()
        i, j, k = self.nn_finder.find(lon, lat, z)
        if use_source_coordinates:
            out_lat = self.nn_finder.lat[i, j]
            out_lon = self.nn_finder.lon[i, j]
            out_z = self.nn_finder.z[k]
        else:
            out_lon = lon
            out_lat = lat
            out_z = z
        cube_list = iris.cube.CubeList()
        for filename in self.file_list:
            print('Reading file {:}'.format(filename))
            ncvar = None
            with netCDF4.Dataset(filename) as f:
                for name in ['standard_name', 'long_name']:
                    for vname in f.variables:
                        v = f[vname]
                        if (hasattr(v, name) and getattr(v, name) == var):
                            ncvar = v
                            break
                    if ncvar is not None:
                        break
                assert ncvar is not None, \
                    'Variable {:} not found in {:}'.format(var, filename)
                if self.nn_finder.data_dim == 3 and len(ncvar.shape) == 4:
                    values = ncvar[:, k, j, i]
                else:
                    values = ncvar[:, j, i]
                units = ncvar.units
                long_name = ncvar.long_name
                timevar = f['time_centered']
                time_array = timevar[:]
                time_units = cf_units.Unit(timevar.units,
                                           calendar=timevar.calendar)
                # convert to a Gregorian calendar
                new_time_units = cf_units.Unit(
                    'seconds since 1970-01-01 00:00:00-00',
                    calendar='gregorian')
                start_date = time_units.num2date(time_array[0])
                offset = new_time_units.date2num(start_date) - time_array[0]
                time_array += offset
                time_dim = iris.coords.DimCoord(
                    time_array,
                    standard_name=timevar.standard_name,
                    units=new_time_units
                )
            # create Cube object
            lon_dim = iris.coords.DimCoord(out_lon, standard_name='longitude',
                                           units='degrees')
            lat_dim = iris.coords.DimCoord(out_lat, standard_name='latitude',
                                           units='degrees')
            dep_dim = iris.coords.DimCoord(-out_z, standard_name='depth',
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
        if dataset_id is not None:
            output.attributes['dataset_id'] = dataset_id
        # make sure we comply with the required metadata policy
        utility.assert_cube_metadata(output)
        return output


def fix_cube_coordinates(cube):
    """
    Fixes NEMO lat,lon coordinates to format that iris supports

    Changes the Cube object in-place.

    :arg cube: a Cube object representing a 2D or 3D NEMO output field.
    """
    # lat,lon coordinates are stored in 2D array which iris does not understand
    # convert coordinates to 1D lat, lon dimensions

    def _make_dim_coord(name, target_len):

        array = cube.coord(name).points
        if numpy.ma.is_masked(array):
            array = array.filled(numpy.nan)
        array = numpy.unique(array)
        if array[0] == 0.0:
            # remove spurios zero coord
            array = array[1:]
        if len(array) == target_len + 1:
            # this should not happen
            # try to compute cell means
            array = 0.5 * (array[1:] + array[:-1])

        assert len(array) == target_len
        dim_coord = iris.coords.DimCoord(array, standard_name=name,
                                         units='degrees')
        return dim_coord

    # FIXME get the coord indices from the metadata
    lat_len, lon_len = cube.coord('latitude').shape
    lon_coord = _make_dim_coord('longitude', lon_len)
    lat_coord = _make_dim_coord('latitude', lat_len)

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
    if len(lat_coord.points) > 1:
        cube.coord('latitude').guess_bounds()
    if len(lon_coord.points) > 1:
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

    # convert time coordinate
    time_coord = cube.coords()[0]
    time_units = time_coord.units
    time_array = numpy.array(time_coord.points)
    start_date = time_units.num2date(time_array[0])

    new_time_units = cf_units.Unit(
        'seconds since 1970-01-01 00:00:00-00',
        calendar='gregorian')

    offset = new_time_units.date2num(start_date) - time_array[0]
    time_array += offset
    time_dim = iris.coords.DimCoord(time_array,
                                    standard_name='time',
                                    units=new_time_units)
    time_ix = cube.coord_dims('time')
    cube.remove_coord(time_coord)
    cube.add_dim_coord(time_dim, time_ix)


def load_nemo_output(ncfile, standard_name, var_name=None, **kwargs):
    """
    Load a field identified with standard_name from NEMO output file.

    Replaces 2D lat,lon coordinate arrays with 1D arrays.

    :arg ncfile: netCDF file name to read
    :arg standard_name: CF standard_name of a field to read
    :returns: an iris Cube object
    """
    cube_list = iris.load(ncfile, standard_name, **kwargs)
    assert len(cube_list) > 0, 'No field {:} found in {:}'.format(
        standard_name, ncfile)
    if var_name is not None:
        new_list = [c for c in cube_list if c.var_name == var_name]
        cube_list = iris.cube.CubeList(new_list)

    assert len(cube_list) == 1, 'Multiple fields found'
    cube = cube_list[0]
    fix_cube_coordinates(cube)
    # read data to memory
    cube.data
    for c in cube.coords():
        c.points
    return cube


def concatenate_nemo_station_data(search_pattern, dataset_id, var_list):
    """
    Reads Nemo4.0 station files and stores as contiquous time series

    :arg str search_pattern: pattern where stations files are located, e.g.,
        '../run_201*/station_*.nc'
    :arg str dataset_id: human readable label for the dataset, e.g.,
        'myrun002'
    :arg var_list: list of variables to store, e.g. ['temp', 'psal', 'slev']
    """

    nreader = NemoStationFileReader(search_pattern,
                                    dataset_id=dataset_id,
                                    verbose=True)
    for var in var_list:
        nemo_var = map_nemo_standard_name[utility.map_var_standard_name[var]]
        var_name = nemo_ncvar_name.get(var)
        dataset = nreader.get_dataset(nemo_var, var_name=var_name)

        for key in dataset:
            cube = dataset[key]

            coords = [c.name() for c in cube.coords()]
            if 'depth' not in coords:
                # assume already at correct depth
                dep_dim = iris.coords.DimCoord(0.0,
                                               standard_name='depth',
                                               units='m')
                cube.add_aux_coord(dep_dim, None)
            utility.save_cube(cube)
