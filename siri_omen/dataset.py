"""
Generic dataset implementation for loading, filtering and matching cubes.
"""
import numpy
import glob
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict
from . import utility


def read_dataset(dataset_id, datatype, variable,
                 location_name=None,
                 start_time=None, end_time=None, verbose=False):
    """
    Read files using pattern

    obs/timeseries/Marviken/temp/ts_Marviken_d0.00m_obs_temp_2016-06-01_2018-10-31.nc

    dataset_id      obs      : observation
    datatype        timeseries
    variable        temp     : sea water temperature
    location_name   Marviken :
    datatype        ts       : time series type
    depth_str       d0.00m   : z=0.00 m

    :arg str dataset_id: dataset identifier, e.g. obs or model run tag
    :arg str variable: variable to read
    :returns: cubes in a dictionary:

    key : location_name-depth_str

    output['Marviken-d0.00m'] = cube
    """
    d = {}
    if location_name is None:
        location_name = '*'
        print('Reading dataset: {:} {:} {:}'.format(
            dataset_id, datatype, variable))
    else:
        print('Reading dataset: {:} {:} {:} {:}'.format(
            dataset_id, datatype, variable, location_name))
    pattern = '{dataset_id:}/{datatype:}/{location_name:}/{var:}/*.nc'.format(
        dataset_id=dataset_id, datatype=datatype,
        location_name=location_name, var=variable)
    if verbose:
        print('Search pattern: {:}'.format(pattern))
    file_list = glob.glob(pattern)
    assert len(file_list) > 0, 'No files found: {:}'.format(pattern)
    for f in sorted(file_list):
        try:
            if verbose:
                print('Loading: {:}'.format(f))
            c = utility.load_cube(f, None)
            if start_time is not None or end_time is not None:
                c = utility.constrain_cube_time(c, start_time=start_time,
                                                end_time=end_time)
            if datatype in ['timeseries', 'timeprofile']:
                dep_str = utility.get_depth_sring(c)
                key = '-'.join((c.attributes['location_name'], dep_str))
            else:
                start_str = utility.get_cube_datetime(c, 0).strftime('%Y-%m-%d')
                key = '-'.join((c.attributes['location_name'], start_str))
            d[key] = c
        except Exception as e:
            print('Could not read file: {:}'.format(f))
            print(e)
    return d


def find_station_pairs(*dataset_list, dist_threshold=0.1,
                       time_overlap=True,
                       unique_pairs=True,
                       time_threshold=None,
                       match_loc_name=False,
                       verbose=False):
    """
    Finds coinciding stations in the given datasets.

    """
    assert len(dataset_list) >= 2, 'at least two dataset are needed'

    def get_loc_coords(dataset):
        keys = list(dataset.keys())
        lon = numpy.array(
            [dataset[k].coord('longitude').points[0] for k in keys]
        )
        lat = numpy.array(
            [dataset[k].coord('latitude').points[0] for k in keys]
        )
        return keys, lon, lat

    # build search tree for the first data set
    src_dataset = dataset_list[0]
    keys, lon, lat = get_loc_coords(src_dataset)
    tree = KDTree(numpy.vstack((lon, lat)).T)

    # store all matching pairs in a dict
    pairs = defaultdict(dict)
    for dataset in dataset_list[1:]:
        args = get_loc_coords(dataset)
        for qkey, qlon, qlat in zip(*args):
            if verbose:
                print('query: {:}'.format(qkey))
            dist_list, ix_list = tree.query([qlon, qlat], k=80)
            for dist, ix in zip(dist_list, ix_list):
                if ix >= len(src_dataset):
                    continue
                if verbose:
                    print(' candidate: {:}'.format(keys[ix]))
                src_key = keys[ix]
                src_loc = src_dataset[src_key].attributes['location_name']
                loc = dataset[qkey].attributes['location_name']
                same_loc_name = src_loc == loc
                match = same_loc_name if match_loc_name \
                    else dist < dist_threshold
                if match:
                    src_cube = src_dataset[src_key]
                    paired_cube = dataset[qkey]
                    overlap = utility.check_cube_overlap(src_cube, paired_cube)
                    if time_overlap and not overlap:
                        continue
                    if isinstance(paired_cube.data, numpy.ma.MaskedArray) \
                            and numpy.all(paired_cube.data.mask):
                        # if all data is bad, skip
                        continue
                    if time_threshold is not None:
                        src_time = utility.get_cube_datetime(src_cube, 0)
                        pair_time = utility.get_cube_datetime(paired_cube, 0)
                        time_diff = abs((src_time - pair_time).total_seconds())
                        if time_diff > time_threshold:
                            continue
                    print('Found pair {:}->{:}, dist={:.2f}'.format(
                        keys[ix], qkey, dist))
                    src_id = src_cube.attributes['dataset_id']
                    pair_id = paired_cube.attributes['dataset_id']
                    if unique_pairs:
                        # add every pair separately
                        key = qkey + ':' + src_key
                    else:
                        # add all entries that match with the query
                        key = src_key
                    pairs[key][src_id] = src_cube
                    pairs[key][pair_id] = paired_cube
    return pairs
