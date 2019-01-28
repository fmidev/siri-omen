import iris  # NOQA
import netCDF4  # NOQA
import numpy  # NOQA
import glob  # NOQA
import collections  # NOQA
import datetime  # NOQA
import matplotlib  # NOQA
import matplotlib.pyplot as plt  # NOQA
import cf_units  # NOQA

from .utility import *  # NOQA
from .dataset import *  # NOQA
from .extract_timeseries import *  # NOQA
from .cmems_reader import *  # NOQA
from .plot_timeseries import *  # NOQA
from .plot_profile import *  # NOQA
from .plot_taylor_diag import *  # NOQA
from .plot_target_diag import *  # NOQA
from .plot_map import *  # NOQA

from . import nemo_reader  # NOQA
from . import cmems_reader  # NOQA
from . import ices_reader  # NOQA
from . import statistics  # NOQA

from distutils.version import StrictVersion
if StrictVersion(iris.__version__) < StrictVersion('2.0.0'):
    iris.FUTURE.netcdf_no_unlimited = True
    iris.FUTURE.netcdf_promote = True
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
