"""
Vertical profile plotting routines
"""
import numpy
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os
from . import utility


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
        if numpy.array(xlim).size > 2:
            cur_xlim = ax.get_xlim()
            target_xlim = None
            for lim in xlim:
                if cur_xlim[0] >= lim[0] and cur_xlim[1] <= lim[1]:
                    target_xlim = lim
                    break
            if target_xlim is not None:
                ax.set_xlim(target_xlim)
        else:
            ax.set_xlim(xlim)
    # assuming that y axis is depth (positive downwards)
    ax.invert_yaxis()
    if title is None:
        loc_names = utility.unique([c.attributes['location_name']
                                    for c in cube_list])
        date = utility.get_cube_datetime(cube_list[0], 0)
        date_str = date.strftime('%Y-%m-%d')
        titles = utility.unique(loc_names)
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

    imgfile = utility.generate_img_filename(cube_list, root_dir=output_dir)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
