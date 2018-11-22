"""
Timeseries plotting routines
"""
from .utility import *  # NOQA

import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os


def generate_img_filename(cube_list, root_dir=None):
    """
    Generate a canonical name for a timeseries image file.
    """
    prefix = 'ts'

    var_str = '-'.join(
        unique([map_var_short_name.get(c.standard_name, c.standard_name)
                for c in cube_list])
    )
    loc_str = '-'.join(
        unique([map_var_short_name.get(c.attributes['location_name'],
                                       c.attributes['location_name'])
                for c in cube_list])
    )

    depth_str = '-'.join(unique([get_depth_sring(c) for c in cube_list]))

    def get_datetime(cube, index):
        t = cube.coord('time')
        return t.units.num2date(t.points[index])

    start_time = min([get_datetime(c, 0) for c in cube_list])
    end_time = max([get_datetime(c, -1) for c in cube_list])
    date_str = '_'.join(
        [d.strftime('%Y-%m-%d') for d in [start_time, end_time]])

    imgfile = '_'.join((prefix, loc_str, depth_str, var_str, date_str))
    imgfile += '.png'

    if root_dir is not None:
        create_directory(root_dir)
        imgfile = os.path.join(root_dir, imgfile)

    return imgfile


def plot_timeseries(ax, cube_list, label_attr='dataset_id', time_lim=None,
                    title=None):
    """
    Plots time series objects in the given axes.

    :arg cube_list: list of cube time series objects to plot. Can be any
        iterable container.
    """

    def get_label(cube, attr_name):
        return cube.attributes.get(attr_name)

    for c in cube_list:
        label = get_label(c, label_attr)
        qplt.plot(c, axes=ax, label=label)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    if title is None:
        loc_names = [c.attributes['location_name'] for c in cube_list]
        dep_strs = ['{:.1f} m'.format(
            c.coord('depth').points.mean()) for c in cube_list]
        titles = unique([' '.join(a) for a in zip(loc_names, dep_strs)])
        title = ', '.join(titles).strip()
        ax.set_title(title)
    if time_lim is not None:
        ax.set_xlim(time_lim)


def save_timeseries_figure(cube_list, output_dir=None, **kwargs):
    """
    Makes a default time series plot and saves it to disk.
    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    plot_timeseries(ax, cube_list, **kwargs)

    imgfile = generate_img_filename(cube_list, root_dir=output_dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
