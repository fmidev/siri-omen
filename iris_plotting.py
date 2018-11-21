"""
Iris plotting routines
"""
import matplotlib.pyplot as plt
import iris.quickplot as qplt
from collections import OrderedDict
import os


var_short_name = {
    'sea_water_practical_salinity': 'salt',
}


def unique(input_list):
    """
    Returns unique elements in a list
    """
    return list(OrderedDict.fromkeys(input_list))


def generate_img_filename(cube_list):
    """
    Generate a canonical name for a timeseries image file.
    """
    prefix = 'ts'

    var_str = '-'.join(
        unique([var_short_name.get(c.standard_name, c.standard_name)
                for c in cube_list])
    )
    loc_str = '-'.join(
        unique([var_short_name.get(c.attributes['location_name'], c.attributes['location_name'])
                for c in cube_list])
    )

    def get_datetime(cube, index):
        t = cube.coord('time')
        return t.units.num2date(t.points[index])

    start_time = min([get_datetime(c, 0) for c in cube_list])
    end_time = max([get_datetime(c, -1) for c in cube_list])
    date_str = '_'.join([d.strftime('%Y-%m-%d') for d in [start_time, end_time]])

    imgfile = '_'.join((prefix, loc_str, var_str, date_str)) + '.png'
    return imgfile


def plot_timeseries(ax, cube_list):
    """
    Plots time series objects in the given axes.

    :arg cube_list: list of cube time series objects to plot. Can be any
        iterable container.
    """

    def get_label(cube):
        attr = 'dataset_name'
        if attr in cube.attributes:
            return cube.attributes[attr]
        return None

    for c in cube_list:
        label = get_label(c)
        qplt.plot(c, axes=ax, label=label)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))


def save_timeseries_figure(cube_list, output_dir=None):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    plot_timeseries(ax, cube_list)

    imgfile = generate_img_filename(cube_list)
    if output_dir is not None:
        imgfile = os.path.join(output_dir, imgfile)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)


