"""
Routines for plotting time-dependent vertical transects.
"""
import numpy
import matplotlib.pyplot as plt
import cf_units
import matplotlib
import os
from . import utility
import datetime
from shapely.geometry import LineString
from cartopy.geodesic import Geodesic

__all__ = [
    'plot_timetransect',
    'make_timetransect_plot',
    'save_timetransect_figure',
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


def get_depth_grid(cube):
    coord = cube.coord('depth')
    assert coord.has_bounds()
    x = numpy.vstack((coord.bounds[0, :, 0], coord.bounds[:, :, 1]))
    x[~numpy.isfinite(x)] = 0.0
    return x


def get_transect_grid(cube):
    """
    Compute the along-transect distance array.

    The cube must have longitude, latitude aux coordinates.
    """
    lon = cube.coord('longitude').points
    lat = cube.coord('latitude').points
    lonlat = numpy.vstack((lon, lat)).T
    geo = Geodesic()
    r = geo.inverse(lonlat[:-1], lonlat[1:])
    dist = numpy.array(r[:, 0])
    # center points
    x = numpy.cumsum(dist)
    x = numpy.hstack(([0], x))
    # covert to plot grid, i.e. compute cell boundaries
    cdist = 0.5*(dist[:-1] + dist[1:])
    cdist = numpy.hstack((cdist[[0]], cdist, cdist[[-1]]))
    x_grid = x - 0.5*cdist
    x_grid = numpy.hstack((x_grid, [x_grid[-1] + 0.5*cdist[-1]]))
    return x_grid


def plot_timetransect(cube, time_index, ax, title=None,
                      log_scale=False, symmetric_scale=False,
                      cmap=None, vmin=None, vmax=None, colorbar=True,
                      show_grid=False, **kwargs):
    """
    Plot a single cube in the given axes
    """
    fig = ax.figure
    _cube = utility.crop_invalid_depths(cube)
    if isinstance(time_index, datetime.datetime):
        tolerance = datetime.timedelta(seconds=1)
        _cube = utility.constrain_cube_time(cube,
                                            start_time=time_index - tolerance,
                                            end_time=time_index + tolerance)
        assert sum(_cube.coord('time').shape) == 1
    else:
        i_time = cube.coord_dims('time')
        assert len(i_time) == 1
        i_time = i_time[0]
        filter = [slice(None)] * len(cube.shape)
        filter[i_time] = time_index
        filter = tuple(filter)
        _cube = cube[filter]
    date = utility.get_cube_datetime(_cube, 0)

    z = -get_depth_grid(_cube)

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

    x_along = get_transect_grid(_cube)
    x_along /= 1000.0  # convert to km
    data = _cube.data.copy()
    # duplicate coords for full cell plotting
    z = numpy.repeat(z, 2, axis=1)
    data = numpy.repeat(data, 2, axis=1)[:, :-1]
    x_along = numpy.repeat(x_along, 2)[1:-1]
    kw = {}
    kw.update(kwargs)
    if show_grid:
        kw['edgecolors'] = '0.5'
        kw['linewidth'] = 0.01
    p = ax.pcolormesh(x_along, z, data, vmin=vmin, vmax=vmax,
                      shading='flat', cmap=cmap, norm=norm, **kw)
    depth_coord = _cube.coord('depth')
    ylabel = '{:} [{:}]'.format(depth_coord.name().capitalize(),
                                depth_coord.units)
    if title is None:
        loc = _cube.attributes['location_name']
        data_id = _cube.attributes['dataset_id']
        date_str = '{:%Y-%m-%d %H:%M}'.format(date)
        title = ' '.join([loc, data_id, date_str])
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Distance [km]')
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

    ax.set_facecolor('0.7')
    return p, date


def make_timetransect_plot(cube_list, time_index, **kwargs):
    ncubes = len(cube_list)

    plot_height = 3.5
    fig = plt.figure(figsize=(12, ncubes * plot_height))
    sharey = kwargs.pop('share_y_axis', True)
    ax_list = fig.subplots(ncubes, 1, sharex=True, sharey=sharey)
    if ncubes == 1:
        ax_list = [ax_list]

    if 'vmin' not in kwargs or kwargs['vmin'] is None:
        kwargs['vmin'] = numpy.min([numpy.nanmin(c.data) for c in cube_list])
    if 'vmax' not in kwargs or kwargs['vmax'] is None:
        kwargs['vmax'] = numpy.max([numpy.nanmax(c.data) for c in cube_list])

    date_list = []
    for cube, ax in zip(cube_list, ax_list):
        p, d = plot_timetransect(cube, time_index, ax, **kwargs)
        date_list.append(d)
    for ax in ax_list[:-1]:
        ax.set_xlabel('')

    assert all([(d - date_list[0]).total_seconds() < 1.0 for d in date_list]),\
        'Time stamps differ {:}'.format(date_list)

    return fig, date_list[0]


def save_timetransect_figure(cube_list, time_index, output_dir=None, **kwargs):
    """
    Makes a default transect plot and saves it to disk.
    """
    imgfile = kwargs.pop('filename', None)
    fig, date = make_timetransect_plot(cube_list, time_index, **kwargs)

    if imgfile is None:
        imgfile = utility.generate_img_filename(cube_list, root_dir=output_dir,
                                                start_time=date,
                                                end_time=date)
    dir, filename = os.path.split(imgfile)
    utility.create_directory(dir)

    print('Saving image {:}'.format(imgfile))
    fig.savefig(imgfile, dpi=200, bbox_inches='tight')
    plt.close(fig)
