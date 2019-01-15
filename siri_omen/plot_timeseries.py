"""
Timeseries plotting routines
"""
import numpy
import matplotlib.pyplot as plt
import iris.quickplot as qplt
import os
from . import utility


def plot_timeseries(ax, cube_list, label_attr='dataset_id', time_lim=None,
                    title=None, time_extent=None,
                    start_time=None, end_time=None, **kwargs):
    """
    Plots time series objects in the given axes.

    :arg cube_list: list of cube time series objects to plot. Can be any
        iterable container.
    """

    def get_label(cube, attr_name):
        return cube.attributes.get(attr_name)

    kwargs.setdefault('linewidth', 1.2)
    for c in cube_list:
        label = get_label(c, label_attr)
        if isinstance(c.data, numpy.ma.MaskedArray) \
                and numpy.all(c.data.mask):
            # if all data is bad, skip
            continue
        qplt.plot(c, axes=ax, label=label, **kwargs)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    ax.set_xlim(start_time, end_time)
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    if title is None:
        loc_names = [c.attributes['location_name'] for c in cube_list]
        dep_strs = ['{:.1f} m'.format(
            c.coord('depth').points.mean()) for c in cube_list]
        titles = utility.unique([' '.join(a) for
                                 a in zip(loc_names, dep_strs)])
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

    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    plot_timeseries(ax, cube_list, time_extent=time_extent,
                    start_time=start_time, end_time=end_time, **kwargs)

    imgfile = utility.generate_img_filename(cube_list, root_dir=output_dir,
                                            start_time=start_time,
                                            end_time=end_time)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
