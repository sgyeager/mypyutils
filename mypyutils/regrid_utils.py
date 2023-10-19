import xarray as xr
import numpy as np
import scipy.sparse as sps
import cf_xarray

def remap_camse(ds, dsw, varlst=[]):
    #dso = xr.full_like(ds.drop_dims('ncol'), np.nan)
    dso = ds.drop_dims('ncol').copy()
    lonb = dsw.xc_b.values.reshape([dsw.dst_grid_dims[1].values, dsw.dst_grid_dims[0].values])
    latb = dsw.yc_b.values.reshape([dsw.dst_grid_dims[1].values, dsw.dst_grid_dims[0].values])
    weights = sps.coo_matrix((dsw.S, (dsw.row-1, dsw.col-1)), shape=[dsw.dims['n_b'], dsw.dims['n_a']])
    if not varlst:
        for varname in list(ds):
            if 'ncol' in(ds[varname].dims):
                varlst.append(varname)
        if 'lon' in varlst: varlst.remove('lon')
        if 'lat' in varlst: varlst.remove('lat')
        if 'area' in varlst: varlst.remove('area')
    for varname in varlst:
        shape = ds[varname].shape
        invar_flat = ds[varname].values.reshape(-1, shape[-1])
        remapped_flat = weights.dot(invar_flat.T).T
        remapped = remapped_flat.reshape([*shape[0:-1], dsw.dst_grid_dims[1].values,
                                          dsw.dst_grid_dims[0].values])
        dimlst = list(ds[varname].dims[0:-1])
        dims={}
        coords={}
        for it in dimlst:
            dims[it] = dso.dims[it]
            coords[it] = dso.coords[it]
        dims['lat'] = int(dsw.dst_grid_dims[1])
        dims['lon'] = int(dsw.dst_grid_dims[0])
        coords['lat'] = latb[:,0]
        coords['lon'] = lonb[0,:]
        remapped = xr.DataArray(remapped, coords=coords, dims=dims, attrs=ds[varname].attrs)
        dso = xr.merge([dso, remapped.to_dataset(name=varname)])
    return dso

def add_grid_bounds(ds):
    saveattrs = ds['lon'].attrs
    ds['lon'] = xr.where(ds['lon']<0,ds['lon']+360,ds['lon'])
    ds['lon'] = ds['lon'].assign_attrs(saveattrs).load()
    ds = ds.cf.add_bounds(['lon','lat']).rename({'lat_bounds':'lat_b','lon_bounds':'lon_b'})
    #  range fix
    latb = ds['lat_b']
    if ((latb<-90) | (latb>90)).any():
        saveattrs = ds['lat'].attrs
        latb = xr.where(latb>90,90,latb)
        latb = xr.where(latb<-90,-90,latb)
        ds['lat_b'] = latb
        ds['lat'] = ds['lat'].assign_attrs(saveattrs).load()
    lonb = ds['lon_b']
    if ((lonb<0) | (lonb>360)).any():
        lonb = xr.where(lonb<0,lonb+360,lonb)
        lonb = xr.where(lonb>360,lonb-360,lonb)
        ds['lon_b'] = lonb
    return ds

def add_grid_bounds_POP(ds):
    lon = ds.lon; lat = ds.lat
    lonb = ds.lon_b; latb = ds.lat_b
    dim1 = lonb.dims[0]; dim2 = lonb.dims[1]
    lonb2 = xr.concat([lonb.isel({dim1:0}),lonb],dim=dim1)
    lonb2 = xr.concat([lonb2.isel({dim2:-1}),lonb2],dim=dim2)
    lonb2 = lonb2.rename({dim1:dim1+'_b',dim2:dim2+'_b'})
    tmp = latb.isel({dim1:0}) - (latb.isel({dim1:1}) - latb.isel({dim1:0}))
    latb2 = xr.concat([tmp,latb],dim=dim1)
    latb2 = xr.concat([latb2.isel({dim2:-1}),latb2],dim=dim2)
    latb2 = latb2.rename({dim1:dim1+'_b',dim2:dim2+'_b'})
    ds = ds.drop(['lon','lat','lon_b','lat_b'])
    lon = lon.drop(['lon','lat','lon_b','lat_b'])
    lat = lat.drop(['lon','lat','lon_b','lat_b'])
    lonb2 = lonb2.drop(['lon','lat','lon_b','lat_b'])
    latb2 = latb2.drop(['lon','lat','lon_b','lat_b'])
    ds = ds.assign_coords({'lon':lon,'lat':lat,'lon_b':lonb2,'lat_b':latb2})
    return ds