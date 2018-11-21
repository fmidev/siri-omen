"""
Time series extraction example
"""
import matplotlib.pyplot as plt

import datetime
import numpy
import iris

#-------------------------------------------------------------------------------
# fabricate a dummy (t, z, y, x) data set
lat = numpy.linspace(40, 50, 11)
lon = numpy.linspace(10, 30, 21)
z = numpy.linspace(0, 20, 6)
time = numpy.arange(32, dtype=float)*600.
date_zero = datetime.datetime(2000, 1, 1)
date_epoch = datetime.datetime.utcfromtimestamp(0)
time_epoch = time + (date_zero - date_epoch).total_seconds()
LON, LAT = numpy.meshgrid(lon, lat)
salt = 10*(numpy.sin(LAT/10.)*numpy.sin(LON/20.) + 1)
time_factor = 1.5 + 0.5*numpy.sin(2*numpy.pi/6/3600.*time)
depth_factor = 0.5 + z/20.
salt = depth_factor[:, numpy.newaxis, numpy.newaxis] * salt[numpy.newaxis, :, :]
salt = time_factor[:, numpy.newaxis, numpy.newaxis, numpy.newaxis] * salt[numpy.newaxis, :, :, :]

# create a cube
time_dim = iris.coords.DimCoord(time_epoch, standard_name='time', units='seconds since 1970-01-01 00:00:00-00')
z_dim = iris.coords.DimCoord(z, standard_name='depth', units='m')
lon_dim = iris.coords.DimCoord(lon, standard_name='longitude', units='degrees')
lat_dim = iris.coords.DimCoord(lat, standard_name='latitude', units='degrees')
cube = iris.cube.Cube(salt, standard_name='sea_water_practical_salinity', units='1')
cube.add_dim_coord(time_dim, 0)
cube.add_dim_coord(z_dim, 1)
cube.add_dim_coord(lat_dim, 2)
cube.add_dim_coord(lon_dim, 3)
iris.save(cube, 'salinity.nc')

#plt.pcolormesh(lon, lat, salt[0, :, :].T); plt.colorbar(); plt.show()

#-------------------------------------------------------------------------------
# extract time series with iris

cube = iris.load('salinity.nc', 'sea_water_practical_salinity')[0]
interp = iris.analysis.Nearest(extrapolation_mode='error')
target = [('depth', 8.0), ('latitude', 42.5), ('longitude', 15.5)]
cube_ts = cube.interpolate(target, interp)

import iris.quickplot as qplt
qplt.plot(cube_ts)
plt.show()

#-------------------------------------------------------------------------------
# efficient nearest neighbor extraction

import netCDF4

# find nearest neighbor indices
target_z = 8.0
target_lat = 42.5
target_lon = 15.5

with netCDF4.Dataset('salinity.nc') as ncfile:
    lat = ncfile['latitude'][:]
    lon = ncfile['longitude'][:]
    z = ncfile['depth'][:]
    # find nearest point in each dimension
    # in the general case one should do a 3D search (e.g. with a KDTree)
    i = numpy.abs(target_z - z).argmin()
    j = numpy.abs(target_lat - lat).argmin()
    k = numpy.abs(target_lon - lon).argmin()

# extract time series
with netCDF4.Dataset('salinity.nc') as ncfile:
    time_epoch = ncfile['time'][:]
    # read only one (x, y, z) point from disk
    values = ncfile['sea_water_practical_salinity'][:, i, j, k]

cube_ts2 = iris.cube.Cube(values, standard_name='sea_water_practical_salinity', units='1')
time_dim = iris.coords.DimCoord(time_epoch, standard_name='time', units='seconds since 1970-01-01 00:00:00-00')
cube_ts2.add_dim_coord(time_dim, 0)

qplt.plot(cube_ts2)
plt.show()

assert numpy.allclose(cube_ts.data, cube_ts2.data)
