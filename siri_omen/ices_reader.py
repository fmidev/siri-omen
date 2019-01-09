"""
Tools for reading ICES CTD cast data from cvs files.
"""
import csv
from scipy.spatial import cKDTree as KDTree
import dateutil.parser
import cf_units
import numpy
from collections import defaultdict
import iris
from . import utility

cf_standard_name = {
    'temp': 'sea_water_temperature',
    'psal': 'sea_water_practical_salinity',
    'pres': 'sea_water_pressure',
}

units_dict = {
    'temp': 'degree_C',
    'psal': '1',
    'pres': 'dbar',
}


def split_fieldname(s):
    """Split csv column name to name and unit"""
    if '[' not in s:
        return s, None
    name, unit = s.split('[')
    name = name.strip()
    unit = unit.split(']')[0]
    return name, unit


def to_float(v):
    """Convert a value str to a float"""
    if v == '':
        return numpy.nan
    try:
        return float(v)
    except Exception as e:
        print(e)
        return numpy.nan


def save_cast(cast):
    """Saves CTD cast as a iris Cube object"""

    data = {}

    meta = {}
    # make profile cube
    for field in cast:
        ncname = field.lower().replace(' ', '_').replace('.', '')
        vals = cast[field]
        fill_value = vals.fill_value if numpy.ma.is_masked(vals) else 1e20
        if ncname == 'time':
            time_units = cf_units.Unit(
                'seconds since 1970-01-01 00:00:00-00',
                calendar='gregorian')
            time_coord = iris.coords.DimCoord(vals, standard_name='time',
                                              units=time_units)
        elif ncname == 'latitude':
            lat_dim = iris.coords.DimCoord(vals, standard_name='latitude',
                                           units='degrees')
        elif ncname == 'longitude':
            lon_dim = iris.coords.DimCoord(vals, standard_name='longitude',
                                           units='degrees')
        elif ncname == 'depth':
            dep_dim = iris.coords.DimCoord(vals, standard_name='depth',
                                           units='m')
        elif isinstance(vals, numpy.ndarray):
            standard_name = cf_standard_name[ncname]
            units = units_dict[ncname]
            data[ncname] = (vals, standard_name, units)
        else:
            meta[ncname] = vals

    for ncname in data:
        vals, standard_name, units = data[ncname]
        cube = iris.cube.Cube(vals,
                              standard_name=standard_name,
                              var_name=ncname,
                              units=units)
        cube.add_dim_coord(dep_dim, 0)
        cube.add_aux_coord(time_coord, None)
        cube.add_aux_coord(lat_dim, None)
        cube.add_aux_coord(lon_dim, None)

        for key, value in meta.items():
            cube.attributes[key] = value

        utility.save_cube(cube)


class CSVStationSet():
    """
    List of all Helcom stations and their locations
    """
    def __init__(self, csv_filename):
        self.station_names = []
        self.station_coordinates = []
        self.csv_filename = csv_filename
        with open(self.csv_filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.station_names.append(row['Station'])
                self.station_coordinates.append(
                    (float(row['Latitude']), float(row['Longitude'])))
        # construct kdtree for searching nearest neighbors
        self.station_coordinates = numpy.array(self.station_coordinates)
        self.tree = KDTree(self.station_coordinates)

    def find_nearest_station(self, lat, lon, tolerance=0.1):
        """
        Finds nearest station in the Helcom list, if any
        """
        dist, index = self.tree.query([[lat, lon]], k=1)
        if dist[0] > tolerance:
            return None, None
        return self.station_names[index[0]], self.station_coordinates[index[0]]

    def get_coordinates(self, station_name):
        """
        Returns lat,lon coordinates of a given station
        """
        ix = self.station_names.index(station_name)
        return self.station_coordinates[ix]


def process_cvs_file(cvs_filename, station_file, dataset_id):
    """
    Process a CVS CTD observation file

    All individual vertical profiles are detected from the data and stored to
    files like
    ctdcast/F33/ctdcast_F33_2017-02-01.nc
    """
    helcom_stations = CSVStationSet(station_file)

    print('Reading file {:}'.format(cvs_filename))

    # detect each cast based on a unique identifier
    key_fields = ['Cruise', 'Station', 'yyyy-mm-ddThh:mm']

    data = defaultdict(list)
    # read data into nested dict: data[key]['PRES']
    with open(cvs_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # construct identifier for each cast
            key = '_'.join([row[f] for f in key_fields])
            d = {}
            for f in row:
                name, unit = split_fieldname(f)
                value = row[f] if row[f] is not None else numpy.nan
                d[name] = value
            data[key].append(d)

    # get all fieldnames and their units
    fieldnames = [split_fieldname(f) for f in reader._fieldnames]
    parsed_units = dict(fieldnames)
    parsed_units['depth'] = 'm'
    fieldnames = [f[0] for f in fieldnames]

    cast_data = {}

    # construct a single cast for each matching record
    for key in data:
        list_dict = defaultdict(list)
        # append all values in a list
        for entry in data[key]:
            for field in entry:
                list_dict[field].append(entry[field])
        cast = {}
        # convert lists to arrays
        for field in list_dict:
            if len(set(list_dict[field])) == 1:
                # single value, purge list
                value = list_dict[field][0]
                if value not in ['', numpy.nan]:
                    if field in ['Latitude', 'Longitude', 'Bot.depth', 'Secchi Depth', 'PRES']:
                        value = float(value)
                    cast[field] = value
            else:
                # make a numpy array
                vals = list_dict[field]
                arr = numpy.array([to_float(v) for v in vals])
                arr = numpy.ma.masked_invalid(arr)
                if not numpy.isnan(arr).all():
                    cast[field] = arr
        # convert time to datetime and epoch
        date_str = cast.pop('yyyy-mm-ddThh:mm')
        date = dateutil.parser.parse(date_str + ' UTC')
        # cast['datetime'] = date
        cast['date'] = date_str
        cast['time'] = utility.datetime_to_epoch(date)
        # try to find a standard station name
        lat = cast['Latitude']
        if isinstance(lat, numpy.ndarray):
            lat = numpy.mean(lat)
        lon = cast['Longitude']
        if isinstance(lon, numpy.ndarray):
            lon = numpy.mean(lon)
        name, _ = helcom_stations.find_nearest_station(lat, lon, 0.05)
        if name is None:
            # Store only Helcom stations
            continue
        cast['location_name'] = name if name is not None else cast['Station']
        cast['dataset_id'] = dataset_id
        # compute depth: 1 db = 1 m depth
        dep = cast['PRES']
        if not isinstance(dep, numpy.ndarray):
            dep = numpy.array([dep])
        cast['depth'] = dep
        cast_data[key] = cast

    for key in cast_data:
        try:
            cast = cast_data[key]
            valid = True
            valid *= len(cast['depth']) > 2
            valid *= numpy.isfinite(cast['Latitude'])
            valid *= numpy.isfinite(cast['Longitude'])
            valid *= numpy.isfinite(cast['time'])
            valid *= cast['location_name'] is not None
            if valid:
                save_cast(cast)
        except Exception as e:
            print('Saving failed: {:}'.format(key))
            print(e)
