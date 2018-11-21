"""
Batch extract example
"""
from siri_omen import *

run_id = 'nemo4-run013'

obs_search_pattern = 'obs/ts_*ssh*.nc'
source_file_pattern = 'NORDIC_1h_20180901_20181029_SURF_grid_T_2018091*.nc'

#extract_timeseries_from_obs(run_id, source_file_pattern, obs_search_pattern)

obs_search_pattern = 'obs/ts_*salt*.nc'
source_file_pattern = 'NORDIC_1h_20180901_20181029_grid_T_2018091*.nc'

extract_timeseries_from_obs(run_id, source_file_pattern, obs_search_pattern)
