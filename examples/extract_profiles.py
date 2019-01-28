"""
Read NEMO timeprofiles and individual profiles at observation time stamps.
"""
from siri_omen import *

var_list = ['temp', 'psal']
obs_id = 'ices-ctd'
dataset_id = 'run001'

nemo_standard_name = {
    'temp': 'sea_water_potential_temperature',
    'salt': 'sea_water_practical_salinity',
}

nemo_var_name = {
    'ssh': 'SSH_inst',
}

ices_standard_name = {
    'temp': 'sea_water_temperature',
    'salt': 'sea_water_salinity',
}


def extract_profile(model_profile, obs_profile, interpolate_depth=False):
    """
    Extract a single profile from a Nemo model based on a obs profile

    Model profile is 4D: [time, depth, latitude, longitude]

    Observation profile is 1D: [depth]
        with scalar variables time, latitude, longitude
    """
    # drop singular lat,lon coords
    m = drop_singleton_dims(model_profile)

    # interpolate in time
    time_var = obs_profile.coord('time')
    target_date = time_var.units.num2date(time_var.points[0])
    target = [('time', target_date)]
    scheme = iris.analysis.Linear(extrapolation_mode='mask')
    m_prof = m.interpolate(target, scheme)

    def _drop_bad_values(c):
        valid_ix = numpy.isfinite(c.data)
        if numpy.ma.is_masked(c.data):
            valid_ix *= ~c.data.mask
        return m_prof[valid_ix]

    m_prof = _drop_bad_values(m_prof)

    if interpolate_depth:
        # interpolate in vertical
        target = [('depth', obs_profile.coord('depth').points)]
        scheme = iris.analysis.Linear(extrapolation_mode='mask')
        m_prof = m_prof.interpolate(target, scheme)
        m_prof = _drop_bad_values(m_prof)

    return m_prof


for var in var_list:
    dataset = read_dataset(dataset_id, 'timeprofile', var)
    obs_dataset = read_dataset(obs_id, 'profile', var)

    pairs = find_station_pairs(obs_dataset, dataset)

    for key in pairs:
        o = pairs[key][obs_id]
        m = pairs[key][dataset_id]

        cube = extract_profile(m, o)
        save_cube(cube)
