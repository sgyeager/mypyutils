# routines for dealing with the calendar e.g., calculating timeseries of seasonal means
import xarray as xr
import numpy as np
import pandas as pd
from math import nan
import cftime
import cf_units


def season_ts(ds, season, var=None):
    """ calculate timeseries of seasonal averages
    Args: ds (xarray.Dataset): dataset
          var (str): variable to calculate 
          season (str): 'DJF', 'MAM', 'JJA', 'SON' or 'DJFM'
          
          Author: I. Simpson
    """

    if (season == 'DJFM'):
        ds_season = ds.where(
        (ds['time.month'] == 12) | (ds['time.month'] == 1) | (ds['time.month'] == 2) | (ds['time.month'] == 3))
        if (var):
            ds_season = ds_season[var].rolling(min_periods=4, center=True, time=4).mean().dropna("time", how="all")
        else:
            ds_season = ds_season.rolling(min_periods=4, center=True, time=4).mean().dropna("time", how="all")

    else:

        ## set months outside of season to nan
        ds_season = ds.where(ds['time.season'] == season)
    
        # calculate 3month rolling mean (only middle months of season will have non-nan values)
        if (var):
            ds_season = ds_season[var].rolling(min_periods=3, center=True, time=3).mean().dropna("time", how='all')
        else:
            ds_season = ds_season.rolling(min_periods=3, center=True, time=3).mean().dropna("time", how="all")

    return ds_season


def time_set_mid(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with midpoints of
    ds[time_name].attrs['bounds'], if bounds attribute exists.
    Except for time_name, the returned Dataset is a copy of ds2.
    The copy is deep or not depending on the argument deep.
    
    Author: K. Lindsay
    """

    ds_out = ds.copy(deep)

    if "bounds" not in ds[time_name].attrs:
        return ds_out

    tb_name = ds[time_name].attrs["bounds"]
    tb = ds[tb_name]
    bounds_dim = next(dim for dim in tb.dims if dim != time_name)

    # Use da = da.copy(data=...), in order to preserve attributes and encoding.

    # If tb is an array of datetime objects then encode time before averaging.
    # Do this because computing the mean on datetime objects with xarray fails
    # if the time span is 293 or more years.
    #     https://github.com/klindsay28/CESM2_coup_carb_cycle_JAMES/issues/7
    if tb.dtype == np.dtype("O"):
        units = "days since 0001-01-01"
        calendar = "noleap"
        tb_vals = cftime.date2num(ds[tb_name].values, units=units, calendar=calendar)
        tb_mid_decode = cftime.num2date(
            tb_vals.mean(axis=1), units=units, calendar=calendar
        )
        ds_out[time_name] = ds[time_name].copy(data=tb_mid_decode)
    else:
        ds_out[time_name] = ds[time_name].copy(data=tb.mean(bounds_dim))

    return ds_out


def time_set_midmonth(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with mid-month
    values (day=15) rather than end-month values.
    
    Author: S. Yeager
    """

    #ds_out = ds.copy(deep)
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    year = xr.where(month==1,year-1,year)
    month = xr.where(month==1,12,month-1)
    nmonths = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], 15) for i in range(nmonths)]
    ds[time_name] = newtime
    return ds

def time_set_midday(ds, time_name, deep=False):
    """
    Return copy of ds with values of ds[time_name] replaced with mid-day
    values (hour=12) rather than end-day values.
    
    Author: S. Yeager
    """
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    day = ds[time_name].dt.day
    nt = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], day[i],12,0,0) for i in range(nt-1)]
    newtime.insert(0,cftime.DatetimeNoLeap(year[0], month[0], day[0]-1,12,0,0))
    ds[time_name] = newtime
    return ds

def cftime_add_yearoffset(ds, time_name, yoff):
    year = ds[time_name].dt.year
    month = ds[time_name].dt.month
    year = xr.where(month==1,year-1,year)
    month = xr.where(month==1,12,month-1)
    nmonths = len(month)
    newtime = [cftime.DatetimeNoLeap(year[i], month[i], 15) for i in range(nmonths)]
    ds[time_name] = newtime
    return ds


def mon_to_seas_old(da):
    """ Converts an Xarray DataArray containing monthly data to one containing 
    seasonal-average data, appropriately weighted with days_in_month. Time coordinate
    of output reflects (approximate) centered time value for DJF, MAM, JJA, SON
    averages.
    
    Author: S. Yeager
    """
    month_length = da.time.dt.days_in_month
    result = ((da * month_length).resample(time='QS-DEC',loffset='45D').sum(skipna=True,min_count=3) /
          month_length.resample(time='QS-DEC',loffset='45D').sum())
    return result

def mon_to_seas(ds):
    # do a simple 3-month rolling mean along L-dimension
    ds_seas = ds.rolling(time=3,min_periods=3, center=True).mean()
    # subselect seasons:  DJF, MAM, JJA, SON
    mon = ds_seas.time.dt.month.values
    timekeep = ((mon == 1) | (mon == 4) | (mon == 7) | (mon == 10))
    ds_seas = ds_seas.isel(time=timekeep)
    return ds_seas

def mon_to_seas_dask(ds):
    """ Converts a Dask DataSet containing monthly data to one containing 
    seasonal-average data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    # drop time(Y,L) variable for now
    ds_seas = ds.drop('time')
    # do a simple 3-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=3,min_periods=3, center=True).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    # subselect seasons:  DJF, MAM, JJA, SON
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where((mon == 1) | (mon == 4) | (mon == 7) | (mon == 10)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

def mon_to_seas_dask2(ds):
    """ Converts a Dask DataSet containing monthly data to one containing 
    seasonal-average data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    # drop time(Y,L) variable for now
    ds_seas = ds.drop('time')
    # do a simple 3-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=3,min_periods=3, center=True).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    # subselect seasons:  JFM, AMJ, JAS, OND
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where((mon == 2) | (mon == 5) | (mon == 8) | (mon == 11)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

def mon_to_ann_dask(ds):
    """ Converts a Dask DataSet containing monthly data to one containing 
    annual-average data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    nyear = int(ds.sizes["L"]/12)
    year = ds.isel(L=slice(0,nyear)).L
    dscat = []
    for i in range(nyear):
        m0 = i*12+1; m1 = m0+11
        dsann = ds.sel(L=slice(m0,m1)).mean('L')
        dsann['time'] = ds.time.sel(L=slice(m0,m1)).mean('L')
        dscat.append(dsann)
    dsout = xr.concat(dscat,year)
    return dsout

def mon_to_3monthseason(ds,monkeep=(np.arange(12)+1)):
    # drop time(Y,L) variable for now
    ds_seas = ds.drop('time')
    # do a simple 3-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=3,min_periods=3, center=True).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where(np.isin(mon,monkeep)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

def mon_to_DJFM(ds):
    """ Converts a Dask DataSet containing monthly data to one containing winter 
    season data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    # drop time(Y,L) variable for now
    ds_seas = ds.drop('time')
    # do a simple 4-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=4,min_periods=4, center=False).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    # select DJFM average
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where((mon==3)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

def mon_to_JJAS(ds):
    """ Converts a Dask DataSet containing monthly data to one containing summer 
    season data. Dask Dataset is assumed to be in initialized-prediction
    format, with dimensions (Y,L,M,...). Time should be a variable, not a coordinate.
    """
    # drop time(Y,L) variable for now
    ds_seas = ds.drop('time')
    # do a simple 4-month rolling mean along L-dimension
    ds_seas = ds_seas.rolling(L=4,min_periods=4, center=False).mean()
    # add time back into dataset
    ds_seas['time'] = ds['time']
    # select DJFM average
    mon = ds_seas.isel(Y=0).time.dt.month.values
    L = ds_seas.L
    Lkeep = L.where((mon==9)).dropna('L')
    ds_seas = ds_seas.sel(L=Lkeep)
    return ds_seas

def time_year_plus_frac(ds, time_name):
    """return time variable, as numpy array of year plus fraction of year values"""

    # this is straightforward if time has units='days since 0000-01-01' and calendar='noleap'
    # so convert specification of time to that representation

    # get time values as an np.ndarray of cftime objects
    if np.dtype(ds[time_name]) == np.dtype("O"):
        tvals_cftime = ds[time_name].values
    else:
        tvals_cftime = cftime.num2date(
            ds[time_name].values,
            ds[time_name].attrs["units"],
            ds[time_name].attrs["calendar"],
        )

    # convert cftime objects to representation mentioned above
    tvals_days = cftime.date2num(
        tvals_cftime, "days since 0000-01-01", calendar="noleap"
    )

    return tvals_days / 365.0
