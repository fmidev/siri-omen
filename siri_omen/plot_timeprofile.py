"""
Routines for plotting time-dependent vertical profiles.
"""
import numpy
import matplotlib.pyplot as plt
import cf_units
import matplotlib
import os
from . import utility

__all__ = [
    'plot_timeprofile',
    'make_timeprofile_plot',
    'save_timeprofile_figure',
]

log_scale_vars = [
    'specific_turbulent_kinetic_energy_of_sea_water',
    'specific_turbulent_kinetic_energy_dissipation_in_sea_water',
    'ocean_vertical_heat_diffusivity',
    'ocean_vertical_momentum_diffusivity',
]

symmetric_vars = [
    'sea_water_x_velocity',
    'sea_water_y_velocity',
    'upward_sea_water_velocity',
]


def get_grid(cube, coordname):
    coord = cube.coord(coordname)
    if not coord.has_bounds():
        coord.guess_bounds()
    x = numpy.hstack((coord.bounds[:, 0], coord.bounds[[-1], 1]))
    return x


def get_plot_time(cube):
    epoch_time_units = cf_units.Unit(
        'seconds since 1970-01-01 00:00:00-00', calendar='gregorian')
    time_coord = cube.coord('time')
    time_coord.convert_units(epoch_time_units)
    t_epoch = get_grid(cube, 'time')
    t = matplotlib.dates.epoch2num(t_epoch)
    return t


def plot_timeprofile(cube, ax, title=None,
                     start_time=None, end_time=None,
                     log_scale=False, symmetric_scale=False,
                     cmap=None, vmin=None, vmax=None, colorbar=True):
    """
    Plot sigle cube in given axes.
    """
    fig = ax.figure
    if start_time is not None or end_time is not None:
        _cube = utility.constrain_cube_time(cube,
                                            start_time=start_time,
                                            end_time=end_time)
    else:
        _cube = cube
    z = -get_grid(_cube, 'depth')
    t = get_plot_time(_cube)

    _log_scale = log_scale or _cube.standard_name in log_scale_vars
    _symmetric_scale = symmetric_scale or _cube.standard_name in symmetric_vars

    if vmin is None:
        vmin = _cube.data.min()
    if vmax is None:
        vmax = _cube.data.max()

    if _log_scale:
        vmin = max(vmin, 1e-12)
        vmax = max(vmax, 1e-12)

    if _symmetric_scale:
        abs_lim = max(abs(vmin), abs(vmax))
        vmin = -abs_lim
        vmax = abs_lim

    norm = None
    if _log_scale:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    p = ax.pcolormesh(t, z, _cube.data.T, vmin=vmin, vmax=vmax, cmap=cmap,
                      norm=norm)
    loc = matplotlib.dates.AutoDateLocator()
    fmt = matplotlib.dates.AutoDateFormatter(loc)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(fmt)
    depth_coord = _cube.coord('depth')
    ylabel = '{:} [{:}]'.format(depth_coord.name().capitalize(),
                                depth_coord.units)
    if title is None:
        loc = _cube.attributes['location_name']
        data_id = _cube.attributes['dataset_id']
        title = ' '.join([loc, data_id])
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    if colorbar:
        # create colorbar
        pad = 0.015
        width = 0.02
        pos = ax.get_position().bounds
        x = pos[0] + pos[2] + pad
        cax = fig.add_axes([x, pos[1], width, pos[3]])
        cb = plt.colorbar(p, cax=cax)
        label = '{:} [{:}]'.format(_cube.name().replace('_', ' ').capitalize(),
                                   _cube.units)
        cb.set_label(label)


def make_timeprofile_plot(cube_list, **kwargs):
    ncubes = len(cube_list)

    plot_height = 3.5
    fig = plt.figure(figsize=(12, ncubes * plot_height))
    ax_list = fig.subplots(ncubes, 1, sharex=True, sharey=True)
    if ncubes == 1:
        ax_list = [ax_list]

    if 'vmin' not in kwargs or kwargs['vmin'] is None:
        kwargs['vmin'] = numpy.min([numpy.nanmin(c.data) for c in cube_list])
    if 'vmax' not in kwargs or kwargs['vmax'] is None:
        kwargs['vmax'] = numpy.max([numpy.nanmax(c.data) for c in cube_list])

    for cube, ax in zip(cube_list, ax_list):
        plot_timeprofile(cube, ax, **kwargs)

    return fig


def save_timeprofile_figure(cube_list, output_dir=None, **kwargs):
    """
    Makes a default time profile plot and saves it to disk.
    """
    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    fig = make_timeprofile_plot(cube_list, start_time=start_time,
                                end_time=end_time, **kwargs)

    imgfile = utility.generate_img_filename(cube_list, root_dir=output_dir,
                                            start_time=start_time,
                                            end_time=end_time)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
