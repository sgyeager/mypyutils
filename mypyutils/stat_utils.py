import xarray as xr
import numpy as np
import sys
import cftime
import xskillscore as xs

def cor_ci_bootyears(ts1, ts2, seed=None, nboots=1000, conf=95):
    """ """
    ptilemin = (100.-conf)/2.
    ptilemax = conf + (100-conf)/2.

    if (ts1.size != ts2.size):
        print("The two arrays must have the same size")
        sys.exit()

    if (seed):
        np.random.seed(seed)

    samplesize = ts1.size
    ranu = np.random.uniform(0, samplesize, nboots*samplesize)
    ranu = np.floor(ranu).astype(int)

    bootdat1 = np.array(ts1[ranu])
    bootdat2 = np.array(ts2[ranu])
    bootdat1 = bootdat1.reshape([samplesize, nboots])
    bootdat2 = bootdat2.reshape([samplesize, nboots])
   
 
    bootcor = xr.corr(xr.DataArray(bootdat1), xr.DataArray(bootdat2), dim='dim_0')
    minci = np.percentile(bootcor,ptilemin)
    maxci = np.percentile(bootcor,ptilemax)

    return minci, maxci

def detrend_linear(dat, dim):
    """ linear detrend dat along the axis dim """
    params = dat.polyfit(dim=dim, deg=1)
    fit = xr.polyval(dat[dim], params.polyfit_coefficients)
    dat = dat-fit
    return dat

def remove_drift(da, da_time, y1, y2):
    """
    Function to convert raw DP DataArray into anomaly DP DataArray with leadtime-dependent climatology removed.
    --Inputs--
        da:  Raw DP DataArray with dimensions (Y,L,M,...)
        da_time:  Verification time of DP DataArray (Y,L)
        y1:  Start year of climatology
        y2:  End year of climatology
        
    --Outputs--
        da_anom:  De-drifted DP DataArray
        da_climo:  Leadtime-dependent climatology
    
    Author: E. Maroon (modified by S. Yeager)
    """
    d1 = cftime.DatetimeNoLeap(y1,1,1,0,0,0)
    d2 = cftime.DatetimeNoLeap(y2,12,31,23,59,59)
    masked_period = da.where((da_time>d1) & (da_time<d2))
    da_climo = masked_period.mean('M').mean('Y')
    da_anom = da - da_climo
    return da_anom, da_climo

def remove_annual_drift(da, da_time, y1, y2):
    """
    Function to convert raw DP DataArray into anomaly DP DataArray with leadtime-dependent climatology removed.
    --Inputs--
        da:  Raw DP DataArray with dimensions (Y,L,M,...)
        da_time:  Verification time of DP DataArray (Y,L)
        y1:  Start year of climatology
        y2:  End year of climatology
        
    --Outputs--
        da_anom:  De-drifted DP DataArray
        da_climo:  Leadtime-dependent climatology
    
    Author: E. Maroon (modified by S. Yeager)
    """
    masked_period = da.where((da_time>=y1) & (da_time<=y2))
    da_climo = masked_period.mean('M').mean('Y')
    da_anom = da - da_climo
    return da_anom, da_climo

def leadtime_monthlybias(da, da_time, obs_clim):
    """
    Function to compute leadtime-dependent monthly bias relative to obs_clim. Assumes inputs have
    used the same climatological window.
    --Inputs--
        da:  DP DataArray containing leadtime-dependent climatology (L,...)  [output from remove_drift()]
        da_time:  DP DataArray containing leadtime-dependent month (L)
        obs_clim:  Observed monthly climatology on same grid as da (month,...)
        
    --Outputs--
        da_anom:  DP DataArray of anomalies relative to obs_clim
    
    Author: S. Yeager
    """
    list = []
    L = da.L
    for i in L.data:
        mon = da_time.sel(L=i).data
        list.append(obs_clim.sel(month=mon))
    obs_clim_new = xr.concat(list,dim=L)
    da_anom = da - obs_clim_new
    return da_anom

def leadtime_skill_seas(mod_da,mod_time,obs_da,detrend=False):
    """ 
    Computes a suite of deterministic skill metrics given two DataArrays corresponding to model and observations, which 
    must share the same lat/lon coordinates (if any). Assumes time coordinates are compatible
    (can be aligned). Both DataArrays should represent 3-month seasonal averages (DJF, MAM, JJA, SON).
    
        Inputs
        mod_da: a seasonally-averaged hindcast DataArray dimensioned (Y,L,M,...)
        mod_time: a hindcast time DataArray dimensioned (Y,L). NOTE: assumes mod_time.dt.month
            returns the mid-month of a 3-month seasonal average (e.g., mon=1 ==> "DJF").
        obs_da: an OBS DataArray dimensioned (season,year,...)
    """
    seasons = {1:'DJF',4:'MAM',7:'JJA',10:'SON'}
    corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []; pers_list = []
    # convert L to leadtime values:
    leadtime = mod_da.L - 2
    for i in mod_da.L.values:
        ens_ts = mod_da.sel(L=i).rename({'Y':'time'})
        ens_time_year = mod_time.sel(L=i).dt.year.data
        ens_time_month = mod_time.sel(L=i).dt.month.data[0]
        obs_ts = obs_da.sel(season=seasons[ens_time_month]).rename({'year':'time'})
        ens_ts = ens_ts.assign_coords(time=("time",ens_time_year))
        a,b = xr.align(ens_ts,obs_ts)
        if detrend:
            a = detrend_linear(a,'time')
            b = detrend_linear(b,'time')
        amean = a.mean('M')
        sigobs = b.std('time')
        sigsig = amean.std('time')
        sigtot = a.std('time').mean('M')
        r = xs.pearson_r(amean,b,dim='time')
        rpc = r/(sigsig/sigtot)
        corr_list.append(r)
        rpc_list.append(rpc.where(r>0))
        rmse_list.append(xs.rmse(amean,b,dim='time')/sigobs)
        msss_list.append(1-(xs.mse(amean,b,dim='time')/b.var('time')))
        pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time'))
    corr = xr.concat(corr_list,leadtime)
    pval = xr.concat(pval_list,leadtime)
    rmse = xr.concat(rmse_list,leadtime)
    msss = xr.concat(msss_list,leadtime)
    rpc = xr.concat(rpc_list,leadtime)
    return xr.Dataset({'corr':corr,'pval':pval,'nrmse':rmse,'msss':msss,'rpc':rpc})

def leadtime_skill_mon(mod_da,mod_time,obs_da,detrend=False):
    """ 
    Computes a suite of deterministic skill metrics given two DataArrays corresponding to model and observations, which 
    must share the same lat/lon coordinates (if any). Assumes time coordinates are compatible
    (can be aligned). Both DataArrays should represent monthly averages.
    
        Inputs
        mod_da: a monthly-averaged hindcast DataArray dimensioned (Y,L,M,...)
        mod_time: a hindcast time DataArray dimensioned (Y,L). NOTE: assumes mod_time.dt.month exists.
        obs_da: an OBS DataArray dimensioned (time,...)
    """
    corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []; pers_list = []
    leadtime = mod_da.L 
    for i in mod_da.L.values:
        ens_ts = mod_da.sel(L=i)
        ens_time = mod_time.sel(L=i)
        ens_ts['Y'] = ens_time
        ens_ts = ens_ts.rename({'Y':'time'}).chunk(dict(time=-1))
        ens_time_mon = ens_time.dt.month.data[0]
        monind = obs_da.time.dt.month==ens_time_mon; obs_ts = obs_da.isel(time=monind)
        a,b = xr.align(ens_ts,obs_ts)
        if detrend:
            a = detrend_linear(a,'time')
            b = detrend_linear(b,'time')
        amean = a.mean('M')
        sigobs = b.std('time')
        sigsig = amean.std('time')
        sigtot = a.std('time').mean('M')
        r = xs.pearson_r(amean,b,dim='time')
        rpc = r/(sigsig/sigtot)
        corr_list.append(r)
        rpc_list.append(rpc.where(r>0))
        rmse_list.append(xs.rmse(amean,b,dim='time')/sigobs)
        msss_list.append(1-(xs.mse(amean,b,dim='time')/b.var('time')))
        pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time'))
    corr = xr.concat(corr_list,leadtime)
    pval = xr.concat(pval_list,leadtime)
    rmse = xr.concat(rmse_list,leadtime)
    msss = xr.concat(msss_list,leadtime)
    rpc = xr.concat(rpc_list,leadtime)
    return xr.Dataset({'corr':corr,'pval':pval,'nrmse':rmse,'msss':msss,'rpc':rpc})

def leadtime_skill_seas_resamp(mod_da,mod_time,obs_da,sampsize,N,detrend=False):
    """ 
    Same as leadtime_skill_seas(), but this version resamples the mod_da member dimension (M) to generate
    a distribution of skill scores using a smaller ensemble size (N, where N<M). Returns the 
    mean of the resampled skill score distribution.
    """
    dslist = []
    seasons = {1:'DJF',4:'MAM',7:'JJA',10:'SON'}
    # convert L to leadtime values:
    leadtime = mod_da.L - 2
    # Perform resampling
    if (not N<mod_da.M.size):
        raise ValueError('ERROR: expecting resampled ensemble size to be less than original')
    mod_da_r = xs.resample_iterations(mod_da.chunk(), sampsize, 'M', dim_max=N)
    for l in mod_da_r.iteration.values:
        corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []; pers_list = []
        for i in mod_da.L.values:
            ens_ts = mod_da_r.sel(iteration=l).sel(L=i).rename({'Y':'time'})
            ens_time_year = mod_time.sel(L=i).dt.year.data
            ens_time_month = mod_time.sel(L=i).dt.month.data[0]
            obs_ts = obs_da.sel(season=seasons[ens_time_month]).rename({'year':'time'})
            ens_ts = ens_ts.assign_coords(time=("time",ens_time_year))
            a,b = xr.align(ens_ts,obs_ts)
            if detrend:
                a = detrend_linear(a,'time')
                b = detrend_linear(b,'time')
            amean = a.mean('M')
            sigobs = b.std('time')
            sigsig = amean.std('time')
            sigtot = a.std('time').mean('M')
            r = xs.pearson_r(amean,b,dim='time')
            rpc = r/(sigsig/sigtot)
            corr_list.append(r)
            rpc_list.append(rpc.where(r>0))
            rmse_list.append(xs.rmse(amean,b,dim='time')/sigobs)
            msss_list.append(1-(xs.mse(amean,b,dim='time')/b.var('time')))
            pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time'))
        corr = xr.concat(corr_list,leadtime)
        pval = xr.concat(pval_list,leadtime)
        rmse = xr.concat(rmse_list,leadtime)
        msss = xr.concat(msss_list,leadtime)
        rpc = xr.concat(rpc_list,leadtime)
        dslist.append(xr.Dataset({'corr':corr,'pval':pval,'rmse':rmse,'msss':msss,'rpc':rpc}))
    dsout = xr.concat(dslist,dim='iteration').mean('iteration').compute()
    return dsout

def compute_skill_annual(mod_da,mod_time,obs_da,nyear=1,nleads=1,resamp=0,detrend=False):
    """
    Computes a suite of skill scores for annual data. Option to use xskillscore resampling to
    compute the mean variance of individual member time series ("sigma_total").
    Assumes mod_time and obs_da.time both contain year values.
    """
    corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []
    sigobs_list = []; sigsig_list = []; sigtot_list = []; s2t_list = []
    if (nyear>1):
        obs_ts = obs_da.rolling(time=nyear,min_periods=nyear, center=True).mean().dropna('time')
    else:
        obs_ts = obs_da
    lvals = np.arange(nyear)
    lvalsda = xr.DataArray(np.arange(nleads),dims="L",name="L")
    for i in range(nleads):
        ens_ts = mod_da.isel(L=lvals+i).mean('L').rename({'Y':'time'})
        ens_time_year = mod_time.isel(L=lvals+i).mean('L')
        ens_ts = ens_ts.assign_coords(time=("time",ens_time_year.data))
        a,b = xr.align(ens_ts,obs_ts)
        #b = b - b.mean('time')
        if detrend:
                a = detrend_linear(a,'time')
                b = detrend_linear(b,'time')
        amean = a.mean('M')
        sigobs = b.std('time')
        sigsig = amean.std('time')
        if (resamp>0):
            iterations = resamp
            ens_size = 1
            a_resamp = xs.resample_iterations(a, iterations, 'M', dim_max=ens_size).squeeze()
            sigtot = a_resamp.std('time').mean('iteration')
        else:
            sigtot = a.std('time').mean('M')
        r = xs.pearson_r(amean,b,dim='time')
        rpc = xr.where(r>0,r,0)/(sigsig/sigtot)
        corr_list.append(r)
        rpc_list.append(rpc)
        rmse_list.append(xs.rmse(amean,b,dim='time')/sigobs)
        msss_list.append(1-(xs.mse(amean,b,dim='time')/b.var('time')))
        pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time'))
        sigobs_list.append(sigobs)
        sigsig_list.append(sigsig)
        sigtot_list.append(sigtot)
        s2t_list.append(sigsig/sigtot)
    corr = xr.concat(corr_list,lvalsda)
    pval = xr.concat(pval_list,lvalsda)
    rmse = xr.concat(rmse_list,lvalsda)
    msss = xr.concat(msss_list,lvalsda)
    rpc = xr.concat(rpc_list,lvalsda)
    sigo = xr.concat(sigobs_list,lvalsda)
    sigs = xr.concat(sigsig_list,lvalsda)
    sigt = xr.concat(sigtot_list,lvalsda)
    s2t  = xr.concat(s2t_list,lvalsda)
    return xr.Dataset({'corr':corr,'pval':pval,'rmse':rmse,'msss':msss,'rpc':rpc,'sig_obs':sigo,'sig_sig':sigs,'sig_tot':sigt,'s2t':s2t})

def compute_resampskill_annual(mod_da,mod_time,obs_da,nyear=1,nleads=1,detrend=False,resamp=0,mean=True):
    """
    Computes a suite of skill scores for annual data.
    Assumes mod_time and obs_da.time both contain year values.
    """
    dslist = []
    if (nyear>1):
        obs_ts = obs_da.rolling(time=nyear,min_periods=nyear, center=True).mean().dropna('time')
    else:
        obs_ts = obs_da
    lvals = np.arange(nyear)
    lvalsda = xr.DataArray(np.arange(nleads),dims="L",name="L")
    for l in mod_da.iteration.values:
        corr_list = []; pval_list = []; rmse_list = []; msss_list = []; rpc_list = []
        sigobs_list = []; sigsig_list = []; sigtot_list = []; s2t_list = []
        for i in range(nleads):
            ens_ts = mod_da.sel(iteration=l).isel(L=lvals+i).mean('L').rename({'Y':'time'})
            ens_time_year = mod_time.isel(L=lvals+i).mean('L').data
            ens_ts = ens_ts.assign_coords(time=("time",ens_time_year))
            a,b = xr.align(ens_ts,obs_ts)
            b = b - b.mean('time')
            if detrend:
                a = detrend_linear(a,'time')
                b = detrend_linear(b,'time')
            amean = a.mean('M')
            sigobs = b.std('time')
            sigsig = amean.std('time')
            if (resamp>0):
                iterations = resamp
                ens_size = 1
                a_resamp = xs.resample_iterations_idx(a, iterations, 'M', dim_max=ens_size).squeeze()
                sigtot = a_resamp.std('time').mean('iteration')
            else:
                sigtot = a.std('time').mean('M')
            r = xs.pearson_r(amean,b,dim='time')
            rpc = xr.where(r>0,r,0)/(sigsig/sigtot)
            #rpc = r/(sigsig/sigtot)
            corr_list.append(r)
            #rpc_list.append(rpc.where(r>0))
            rpc_list.append(rpc)
            rmse_list.append(xs.rmse(amean,b,dim='time')/sigobs)
            msss_list.append(1-(xs.mse(amean,b,dim='time')/b.var('time')))
            pval_list.append(xs.pearson_r_eff_p_value(amean,b,dim='time'))
            sigsig_list.append(sigsig)
            sigtot_list.append(sigtot)
            s2t_list.append(sigsig/sigtot)
        corr = xr.concat(corr_list,lvalsda)
        pval = xr.concat(pval_list,lvalsda)
        rmse = xr.concat(rmse_list,lvalsda)
        msss = xr.concat(msss_list,lvalsda)
        rpc = xr.concat(rpc_list,lvalsda)
        sigs = xr.concat(sigsig_list,lvalsda)
        sigt = xr.concat(sigtot_list,lvalsda)
        s2t  = xr.concat(s2t_list,lvalsda)
        dslist.append(xr.Dataset({'corr':corr,'pval':pval,'rmse':rmse,'msss':msss,'rpc':rpc,'sig_sig':sigs,'sig_tot':sigt,'s2t':s2t}))
    dsout = xr.concat(dslist,dim='iteration')
    if (mean):
        dsout = dsout.mean('iteration')
    return dsout
