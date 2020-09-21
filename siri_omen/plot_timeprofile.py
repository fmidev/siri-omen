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

var_short_name = {
    'specific_turbulent_kinetic_energy_of_sea_water': 'tke',
    'specific_turbulent_kinetic_energy_dissipation_in_sea_water': 'tke dissipation rate',
    'ocean_vertical_heat_diffusivity': 'eddy diffusivity',
    'ocean_vertical_momentum_diffusivity': 'eddy viscosity',
}


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
                     label_alias=None,
                     cmap=None, vmin=None, vmax=None, colorbar=True):
    """
    Plot a single cube in the given axes.
    """
    fig = ax.figure
    if start_time is not None or end_time is not None:
        _cube = utility.constrain_cube_time(cube,
                                            start_time=start_time,
                                            end_time=end_time)
    else:
        _cube = cube
    _cube = utility.crop_invalid_depths(_cube)

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

    if cube.attributes['dataset_id'][:5] == 'diff:':
        # this must be a diff field
        cmap = plt.get_cmap('RdBu_r')
        val_max = numpy.nanmax(numpy.abs(cube.data))
        vmin = -val_max
        vmax = val_max

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
        if label_alias is not None:
            data_id = label_alias.get(data_id, data_id)
        title = ' '.join([loc, data_id])
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.autoscale(enable=True, axis='x', tight=True)
    fig.autofmt_xdate()
    if colorbar:
        # create colorbar
        pad = 0.015
        width = 0.02
        pos = ax.get_position().bounds
        x = pos[0] + pos[2] + pad
        cax = fig.add_axes([x, pos[1], width, pos[3]])
        cb = plt.colorbar(p, cax=cax)
        var_name = _cube.name()
        var_name = var_short_name.get(var_name, var_name)
        var_name = var_name.replace('_', ' ').capitalize()
        label = '{:} [{:}]'.format(var_name, _cube.units)
        cb.set_label(label)


def make_timeprofile_plot(cube_list, **kwargs):
    _cube_list = list(cube_list)

    if 'vmin' not in kwargs or kwargs['vmin'] is None:
        kwargs['vmin'] = numpy.min([numpy.nanmin(c.data) for c in _cube_list])
    if 'vmax' not in kwargs or kwargs['vmax'] is None:
        kwargs['vmax'] = numpy.max([numpy.nanmax(c.data) for c in _cube_list])

    plot_diff = kwargs.pop('plot_diff', False)

    if plot_diff:
        # FIXME most hacky
        diff = _cube_list[1] - _cube_list[0]
        diff.attributes['location_name'] = _cube_list[0].attributes['location_name']
        diff.attributes['dataset_id'] = 'diff:{:}-{:}'.format(*[c.attributes['dataset_id'] for c in [_cube_list[1], _cube_list[0]]])
        diff.standard_name = _cube_list[0].standard_name
        diff.long_name = 'diff:' + _cube_list[0].standard_name
        diff.units = _cube_list[0].units
        _cube_list.append(diff)

    ncubes = len(_cube_list)
    plot_height = 3.5
    fig = plt.figure(figsize=(12, ncubes * plot_height))
    sharey = kwargs.pop('share_y_axis', True)
    ax_list = fig.subplots(ncubes, 1, sharex=True, sharey=sharey)
    if ncubes == 1:
        ax_list = [ax_list]

    for cube, ax in zip(_cube_list, ax_list):
        plot_timeprofile(cube, ax, **kwargs)

    return fig


def save_timeprofile_figure(cube_list, output_dir=None, plot_root_dir=None, **kwargs):
    """
    Makes a default time profile plot and saves it to disk.
    """
    time_extent = kwargs.pop('time_extent', None)
    start_time = kwargs.pop('start_time', None)
    end_time = kwargs.pop('end_time', None)
    imgfile = kwargs.pop('filename', None)
    if start_time is None and end_time is None and time_extent is not None:
        start_time, end_time = utility.get_common_time_overlap(cube_list,
                                                               time_extent)
    fig = make_timeprofile_plot(cube_list, start_time=start_time,
                                end_time=end_time, **kwargs)

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
