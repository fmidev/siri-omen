"""
Generic dataset implementation for loading, filtering and matching cubes.
"""
from . import *  # NOQA
from scipy.spatial import cKDTree as KDTree
from collections import defaultdict


def read_dataset(dataset_id, datatype, variable, verbose=False):
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
    print('Reading dataset: {:} {:} {:}'.format(
        dataset_id, datatype, variable))
    d = {}
    pattern = '{dataset_id:}/{datatype:}/{location_name:}/{var:}/*.nc'.format(
        dataset_id=dataset_id, datatype=datatype,
        location_name='*', var=variable)
    if verbose:
        print('Search pattern: {:}'.format(pattern))
    file_list = glob.glob(pattern)
    assert len(file_list) > 0, 'No files found: {:}'.format(pattern)
    for f in file_list:
        if verbose:
            print('Loading: {:}'.format(f))
        c = load_cube(f, None)
        if datatype in ['timeseries', 'timeprofile']:
            dep_str = get_depth_sring(c)
            key = '-'.join((c.attributes['location_name'], dep_str))
        else:
            start_str = get_cube_datetime(c, 0).strftime('%Y-%m-%d')
            key = '-'.join((c.attributes['location_name'], start_str))
        d[key] = c
    return d


def find_station_pairs(*dataset_list, dist_threshold=0.1, time_threshold=None):
    """
    Finds coinciding stations in the given datasets.

    """

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
            dist_list, ix_list = tree.query([qlon, qlat], k=10)
            for dist, ix in zip(dist_list, ix_list):
                if dist < dist_threshold:
                    src_key = keys[ix]
                    src_cube = src_dataset[src_key]
                    paired_cube = dataset[qkey]
                    if isinstance(paired_cube.data, numpy.ma.MaskedArray) \
                            and numpy.all(paired_cube.data.mask):
                        # if all data is bad, skip
                        continue
                    if time_threshold is not None:
                        src_time = get_cube_datetime(src_cube, 0)
                        pair_time = get_cube_datetime(paired_cube, 0)
                        time_diff = abs((src_time - pair_time).total_seconds())
                        if time_diff > time_threshold:
                            continue
                    print('Found pair {:}->{:}, dist={:.2f}'.format(
                        keys[ix], qkey, dist))
                    src_id = src_cube.attributes['dataset_id']
                    pair_id = paired_cube.attributes['dataset_id']
                    pairs[src_key][src_id] = src_cube
                    pairs[src_key][pair_id] = paired_cube
    return pairs
