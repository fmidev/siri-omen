"""
CMEMS read example
"""
from siri_omen import *

search_pattern = '201*/*.nc'
# search_pattern = '201*/BO_201*_TS_MO_Helsinki.nc'
dataset_id = 'cmems-nrt'
start_time = datetime.datetime(2016, 6, 1)
end_time = datetime.datetime(2018, 7, 1)

sname_list = [
    'water_surface_height_above_reference_datum',
    'sea_water_temperature',
    'sea_water_practical_salinity',
]

for standard_name in sname_list:
    import_cmems_timeseries(dataset_id, search_pattern, standard_name,
                            start_time=start_time, end_time=end_time,
                            verbose=True)
