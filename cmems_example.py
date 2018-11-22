"""
CMEMS read example

Reads CMEMS mooring netCDF files and converts them into timeseries files.
"""
from siri_omen import *  # NOQA

search_pattern = '2018*/*.nc'
dataset_id = 'obs'
start_time = datetime.datetime(2018, 9, 1)
end_time = datetime.datetime(2018, 10, 30)

sname_list = [
    'water_surface_height_above_reference_datum',
    'sea_water_temperature',
    'sea_water_practical_salinity',
]
for standard_name in sname_list:
    import_cmems_timeseries(dataset_id, search_pattern, standard_name,
                            start_time=start_time, end_time=end_time,
                            verbose=True)
