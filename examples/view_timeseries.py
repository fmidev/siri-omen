"""
View netCDF time series in an interactive timeseries plot with bokeh.

Plot opens in a browser window.
"""
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category10 as palette
import itertools
import numpy
import argparse
from siri_omen import *
import cftime

color_cycle = itertools.cycle(palette[10])


def make_interactive_plot(cube_list):

    p = figure(x_axis_type='datetime', plot_width=1400, plot_height=700)

    for cube in cube_list:
        time_coord = cube.coord('time')
        time_units = time_coord.units
        time_array = numpy.array(time_coord.points)
        # get cftime calendar-aware datetime
        datetime_list = time_units.num2date(time_array)
        # convert to python datetime
        datetime_list = [
            datetime.datetime(d.year, d.month, d.day, d.hour, d.second) for d
            in datetime_list
        ]
        values = cube.data.flatten()

        legend = cube.attributes['dataset_id']
        p.line(datetime_list, values, legend_label=legend, color=next(color_cycle))

    ylabel = '{:} [{:}]'.format(cube.standard_name.replace('_', ' '),
                                cube.units)
    p.yaxis.axis_label = ylabel
    p.title.text = cube.attributes['location_name']

    output_file('timeseries.html')
    show(p)


def load_cubes(file_list):
    cube_list = []
    var_list = ['slev', 'temp', 'psal']
    for f in file_list:
        c = None
        for v in var_list:
            try:
                c = load_cube(f, v)
                break
            except Exception:
                pass
        assert c is not None, 'Could not open file {:}'.format(f)
        cube_list.append(c)
    return cube_list


def parse_args():

    parser = argparse.ArgumentParser(
        description='Open netCDF time series files in an interactive plot.',
        # includes default values in help entries
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('files', metavar='file', nargs='+',
                        help='a netCDF file to open, multiple values accepted')
    parser.add_argument('--remove-mean', action='store_true',
                        help='remove mean from time series prior to plotting')
    args = parser.parse_args()
    cube_list = load_cubes(args.files)
    if args.remove_mean:
        for c in cube_list:
            c.data -= c.data.mean()
    make_interactive_plot(cube_list)


if __name__ == '__main__':
    parse_args()
