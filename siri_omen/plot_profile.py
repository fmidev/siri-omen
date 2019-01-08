"""
Vertical profile plotting routines
"""
from .utility import *  # NOQA

import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os


def generate_img_filename(cube_list, root_dir=None):
    """
    Generate a canonical name for a vertical profile image file.
    """
    prefix = 'vprof'

    var_str = '-'.join(
        unique([map_var_short_name.get(c.standard_name, c.standard_name)
                for c in cube_list])
    )
    loc_str = '-'.join(
        unique([map_var_short_name.get(c.attributes['location_name'],
                                       c.attributes['location_name'])
                for c in cube_list])
    )

    start_time = [get_cube_datetime(c, 0) for c in cube_list]
    date_str = start_time[0].strftime('%Y-%m-%d')

    imgfile = '_'.join((prefix, loc_str, var_str, date_str))
    imgfile += '.png'

    if root_dir is not None:
        create_directory(root_dir)
        imgfile = os.path.join(root_dir, imgfile)

    return imgfile


def plot_profile(ax, cube_list, label_attr='dataset_id', xlim=None,
                 title=None, **kwargs):
    """
    Plots vertical profile objects in the given axes.

    :arg cube_list: list of cube objects to plot. Can be any
        iterable container.
    """

    def get_label(cube, attr_name):
        return cube.attributes.get(attr_name)

    for c in cube_list:
        label = get_label(c, label_attr)
        if isinstance(c.data, numpy.ma.MaskedArray) \
                and numpy.all(c.data.mask):
            # if all data is bad, skip
            continue
        depth_coord = c.coord('depth')
        qplt.plot(c, depth_coord, axes=ax, label=label, **kwargs)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    if xlim is not None:
        ax.set_xlim(xlim)
    if title is None:
        loc_names = unique([c.attributes['location_name']
                            for c in cube_list])
        date = get_cube_datetime(cube_list[0], 0)
        date_str = date.strftime('%Y-%m-%d')
        titles = unique(loc_names)
        title = ', '.join(titles).strip()
        title += ' ' + date_str
        ax.set_title(title)


def save_profile_figure(cube_list, output_dir=None, **kwargs):
    """
    Makes a default time series plot and saves it to disk.
    """
    fig = plt.figure(figsize=(5, 12))
    ax = fig.add_subplot(111)

    plot_profile(ax, cube_list, **kwargs)

    imgfile = generate_img_filename(cube_list, root_dir=output_dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
