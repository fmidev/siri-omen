# siri-omen | nemo-iris

Post-processing tools for NEMO ocean model outputs.

## Installation

### Install Iris

`siri-omen` uses [Iris](https://scitools.org.uk/iris/docs/latest/) as it's
data model.
    Iris is easiest to install with Anaconda Python.

It recommended to create a specific conda environment for Iris, for example:
```bash
conda create --name iris python=3
conda activate iris
```

The environment must be activated prior usage:
```bash
conda activate iris
```

Then we can install Iris. At the moment it is recommended to use version 1.13.

```bash
conda install -c conda-forge iris=1.13
```

### Install with pip

Once Iris is installed, and the `iris` environment is active, install
`siri-omen` with

```bash
pip install -e /path/to/siri-omen/repo/
```

## Features

- time series plots
- vertical profile plots
- comparison of datasets, interpolation on common grid
- computation of statistics
- Taylor and target diagrams
- Geographical plots (maps)

## Data model

`siri-omen` data model uses Iris Cube objects to represent data
(see [Iris documentation](https://scitools.org.uk/iris/docs/latest/userguide/iris_cubes.html)).
In addition, two metadata entries are required:

1. `cube.attributes['dataset_id']`: A string that identifies the data set,
    e.g. `mynemorun1` or `observations`
2. `cube.attributes['location_name']`: A string that identifices the spatial
    location of the data, e.g. station or transect identifier.

Four different kinds of geospatial data are suported:

1. `point`: pointwise data, data dimensions: `()`
2. `timeseries`: scalar time series data, data dimensions: `('time')`
3. `profile`: vertical profile data, data dimensions: `('depth')`
4. `timeprofile`: time dependent vertical profile data, data dimensions: `('time', 'depth')`

## Reading netCDF data

`siri-omen` can read netCDF files that contain sufficient metadata.
It is recommended to first generate Iris Cube objects and then store them to
disk.

To support reading in various model output files, some metadata editing may be
necssary. See `siri_omen/nemo_reader.py`.
