"""
Routines for plotting time-dependent vertical transects.
"""
import numpy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import os
from . import utility

__all__ = [
    'plot_scatter',
    'make_scatter_plot',
    'save_scatter_figure',
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


def plot_scatter(
        cube, ax, x_coordinate, y_coordinate,
        x_offset=None, y_offset=None,
        title=None,
        log_scale=False, symmetric_scale=False,
        cmap=None, vmin=None, vmax=None, colorbar=True,
        show_grid=False, **kwargs):
    """
    Plot a single cube in the given axes
    """
    fig = ax.figure
    _log_scale = log_scale or cube.standard_name in log_scale_vars
    _symmetric_scale = symmetric_scale or cube.standard_name in symmetric_vars

    if vmin is None:
        vmin = cube.data.min()
    if vmax is None:
        vmax = cube.data.max()

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

    if cube.attributes['dataset_id'][:5] == 'diff:':
        # this must be a diff field
        cmap = plt.get_cmap('RdBu_r')
        val_max = numpy.nanmax(numpy.abs(cube.data))
        vmin = -val_max
        vmax = val_max

    label_alias = kwargs.pop('label_alias', None)

    if x_coordinate == 'time':
        x_coord = cube.coord('time')
        x_coord.convert_units('seconds since 1970-01-01+00')
        x = mdates.epoch2num(x_coord.points)
    else:
        x_coord = cube.coord(x_coordinate)
        x = x_coord.points.copy()
    y_coord = cube.coord(y_coordinate)
    y = y_coord.points.copy()

    def compute_coord_shift(x_offset):
        coord_name, scalar, offset = x_offset
        coord = cube.coord(coord_name)
        if coord_name == 'time':
            coord.convert_units('seconds since 1970-01-01+00')
        coord_array = coord.points
        if offset == 'remove-first':
            b = -coord_array[0]
        elif offset == 'remove-last':
            b = -coord_array[-1]
        else:
            b = offset
        x_shift = scalar*(coord_array + b)
        return x_shift

    if x_offset is not None:
        x += compute_coord_shift(x_offset)
    if y_offset is not None:
        y += compute_coord_shift(y_offset)

    values = cube.data

    kw = {}
    kw.setdefault('alpha', 1.0)
    kw.setdefault('edgecolors', 'none')
    kw.setdefault('s', 10)
    kw.update(kwargs)
    color = kw.pop('c', None)
    if color is None:
        color = values
    p = ax.scatter(x, y, c=color, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                   **kw)

    def get_time_locator():
        xlim = ax.get_xlim()
        range_days = xlim[1] - xlim[0]
        if range_days < 15:
            major_locator = mdates.DayLocator()
            minor_locator = mdates.HourLocator(interval=6)
        elif range_days < 30:
            major_locator = mdates.DayLocator([1, 5, 10, 15, 20, 25])
            minor_locator = mdates.DayLocator()
        elif range_days < 80:
            major_locator = mdates.DayLocator([1, 10, 20])
            minor_locator = mdates.DayLocator()
        elif range_days < 370:
            major_locator = mdates.MonthLocator()
            minor_locator = mdates.DayLocator([1, 5, 10, 15, 20, 25])
        else:
            major_locator = mdates.AutoDateLocator(
                minticks=7, maxticks=12, interval_multiples=False
            )
            minor_locator = mdates.DayLocator([1, 15])
        return major_locator, minor_locator

    if x_coordinate == 'time':
        major_locator, minor_locator = get_time_locator()
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_minor_locator(minor_locator)
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)

    if title is None:
        loc = cube.attributes['location_name']
        data_id = cube.attributes['dataset_id']
        if label_alias is not None:
            data_id = label_alias.get(data_id, data_id)
        title = ' '.join([loc, data_id])
    ax.set_title(title)

    # add x label
    if x_coordinate == 'time':
        ax.set_xlabel('Date')
    else:
        x_str = x_coord.standard_name.replace('_', ' ').capitalize()
        xlabel = f'{x_str} [{x_coord.units}]'
        ax.set_xlabel(xlabel)
    # add y label
    y_str = y_coord.standard_name.replace('_', ' ').capitalize()
    ylabel = f'{y_str} [{y_coord.units}]'
    ax.set_ylabel(ylabel)

    if colorbar:
        # create colorbar
        pad = 0.015
        width = 0.02
        pos = ax.get_position().bounds
        x = pos[0] + pos[2] + pad
        cax = fig.add_axes([x, pos[1], width, pos[3]])
        cb = plt.colorbar(p, cax=cax)
        label = '{:} [{:}]'.format(cube.name().replace('_', ' ').capitalize(),
                                   cube.units)
        cb.set_label(label)

    return p


def make_scatter_plot(cube_list, *args, **kwargs):
    _cube_list = list(cube_list)

    if 'vmin' not in kwargs or kwargs['vmin'] is None:
        kwargs['vmin'] = numpy.min([numpy.nanmin(c.data) for c in _cube_list])
    if 'vmax' not in kwargs or kwargs['vmax'] is None:
        kwargs['vmax'] = numpy.max([numpy.nanmax(c.data) for c in _cube_list])

    plot_diff = kwargs.pop('plot_diff', False)

    if plot_diff:
        # compute difference between first 2 cubes
        [c.data for c in _cube_list]  # force real data (looks like iris bug)
        # first cube is the observation
        a = _cube_list[0].copy()
        b = _cube_list[1].copy()
        if not a.is_compatible(b):
            b = utility.align_cubes(a, b)
        diff = b.copy()
        diff.data = b.data - a.data
        assert numpy.abs(diff.data).max() < 1e10, 'Bad values in diff field'
        location_name = a.attributes['location_name']
        diff.attributes['location_name'] = location_name
        id_list = [c.attributes['dataset_id'] for c in [b, a]]
        diff.attributes['dataset_id'] = 'diff:{:}-{:}'.format(*id_list)
        diff.standard_name = a.standard_name
        diff.long_name = 'diff:' + a.standard_name
        diff.units = a.units
        _cube_list.append(diff)

    ncubes = len(_cube_list)
    plot_height = 4.5
    fig = plt.figure(figsize=(6, ncubes * plot_height))
    sharey = kwargs.pop('share_y_axis', True)
    ax_list = fig.subplots(ncubes, 1, sharex=True, sharey=sharey)
    if ncubes == 1:
        ax_list = [ax_list]

    for cube, ax in zip(_cube_list, ax_list):
        plot_scatter(cube, ax, *args, **kwargs)
    for ax in ax_list[:-1]:
        ax.set_xlabel('')

    return fig, ax_list


def save_scatter_figure(cube_list, *args,
                        output_dir=None, plot_root_dir=None, **kwargs):
    """
    Makes a default scatter plot and saves it to disk.
    """
    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    imgfile = kwargs.pop('filename', None)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    fig, ax_list = make_scatter_plot(
        cube_list, *args, start_time=start_time, end_time=end_time, **kwargs)

    if imgfile is None:
        imgfile = utility.generate_img_filename(cube_list,
                                                output_dir=output_dir,
                                                root_dir=plot_root_dir,
                                                start_time=start_time,
                                                end_time=end_time)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
