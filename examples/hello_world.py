"""
A simple example of time series data processing.
"""
from siri_omen import *


def make_timeseries_cube(time, values, standard_name, units,
                         depth=0.0,
                         time_units='seconds since 1970-01-01 00:00:00-00',
                         calendar='gregorian',
                         attributes=None):
    """
    Create a timeseries Cube object.

    Parameters
    ----------

    time: array_like (ntime, )
        array of time stamps
    values: array_like (ntime, )
        array of values
    standard_name: str
        CF standard name of the variable.
        E.g. 'sea_water_temperature'
    units: srt
        CF standard units of the variable.
        E.g. 'degree_C'
    depth: float
        Depth in meters. Default: 0.0
    time_units: str
        CF standard units of the time variable.
        E.g. 'seconds since 1970-01-01 00:00:00-00'
    calendar: str
        CF standard calendar of the time variable.
        Default: 'gregorian'
    attributes: dict
        Dictionary of metadata attributes.

    Returns
    -------

    Timeseries as a Iris Cube object.

    """
    import iris
    import cf_units
    msg = 'length of time and values arrays does not match.'
    assert len(time) == len(values), msg

    # define time coordinate
    time_units_obj = cf_units.Unit(time_units, calendar=calendar)

    # make time DimCoord
    time_coord = iris.coords.DimCoord(time, standard_name='time',
                                      units=time_units_obj)

    # make a cube
    cube = iris.cube.Cube(values, standard_name=standard_name, units=units)

    # add time coordinate to dimension 0
    cube.add_dim_coord(time_coord, 0)

    # add aux depth coordinate (required for time series)
    dep_coord = iris.coords.DimCoord([depth], standard_name='depth',
                                     units='m')
    cube.add_aux_coord(dep_coord)

    if attributes is not None:
        # add attributes
        cube.attributes.update(attributes)

    return cube

# ------------------------------------------------------------------------------
# fabricate data to represent observations

t = numpy.linspace(0, 24*3600*25, 500)
data = numpy.sin(2*numpy.pi*t/(24*3600))
cube = make_timeseries_cube(
    t, data, 'water_surface_height_above_reference_datum', 'm')

# add required metadata
cube.attributes['dataset_id'] = 'observation'  # string to identify data origin
cube.attributes['location_name'] = 'Marviken'  # string to identify location

# ------------------------------------------------------------------------------
# save cube to disk in canonical format (uses metadata to generate filename)

save_cube(cube)

# ------------------------------------------------------------------------------
# fabricate data to represent a model

data = 0.8*data + 0.1
cube = make_timeseries_cube(
    t, data, 'water_surface_height_above_reference_datum', 'm')
cube.attributes['dataset_id'] = 'model-run-01'
cube.attributes['location_name'] = 'Marviken'
save_cube(cube)

# ------------------------------------------------------------------------------
# load cubes from disk
# 'slev' is a short name for 'water_surface_height_above_reference_datum'

ds_dict = read_dataset('observation', 'timeseries',
                       'slev',
                       location_name='Marviken')
o = ds_dict[list(ds_dict.keys())[0]]
ds_dict = read_dataset('model-run-01', 'timeseries',
                       'slev',
                       location_name='Marviken')
m = ds_dict[list(ds_dict.keys())[0]]

# ------------------------------------------------------------------------------
# crop time in observation data

start_time = datetime.datetime(1970, 1, 5)
end_time = None
o = constrain_cube_time(o, start_time, end_time)

# ------------------------------------------------------------------------------
# make a time series plot

cube_list = [o, m, ]  # list of cubes to plot
save_timeseries_figure(
    cube_list,
    time_extent='union',  # choose how the time limits are chosen
)

# ------------------------------------------------------------------------------
# make taylor-target plot

cube_pairs = [(o, m), ]  # list of (observation, model) pairs to plot
save_taylor_target_diagram(
    cube_pairs,
    label_attr='location_name',  # choose label attrib
)
