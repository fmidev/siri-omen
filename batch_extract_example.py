"""
Batch extract example
"""
from siri_omen import *

run_id = 'nemo4-run013'

source_file_pattern = 'NORDIC_1h_20180901_20181029_SURF_grid_T_2018091*.nc'
obs_search_pattern = 'obs/ts_*ssh*.nc'
extract_timeseries_from_obs(run_id, source_file_pattern, obs_search_pattern)


source_file_pattern = 'NORDIC_1h_20180901_20181029_grid_T_2018091*.nc'
obs_search_pattern = 'obs/ts_*salt*.nc'
extract_timeseries_from_obs(run_id, source_file_pattern, obs_search_pattern)

obs_search_pattern = 'obs/ts_*temp*.nc'
extract_timeseries_from_obs(run_id, source_file_pattern, obs_search_pattern)
